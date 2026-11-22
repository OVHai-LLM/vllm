# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side decode loop for a few greedy requests (engine side).

When every live request is a plain greedy decode (no logprobs, penalties,
logits processors, grammar, LoRA, ...), at most the worker's largest loop
size, and nothing is waiting, the engine schedules one regular decode step
with KV blocks preallocated for each request's whole token budget and asks
the worker to run it, and the following steps, in its device-side loop
(vllm/v1/worker/gpu/decode_loop.py). The GPU then decodes without host round
trips; each engine step polls the loop's host ring and feeds the new tokens
through the regular ``Scheduler.update_from_output`` (stop checks, outputs,
frees, stats), so the frontend sees ordinary outputs. A new request, an
abort or any other host-side finish of a row the GPU still runs makes the
engine ask the GPU to stop after the current iteration; once the loop is over
the engine resumes regular steps (and may start a new loop).

The GPU stops a row by itself at its budget (``max_tokens`` /
``max_model_len``) or at an EOS / stop token id, the same conditions
``check_stop`` applies first, so outputs are the tokens regular steps would
produce. Stop strings and other host-side conditions end the loop instead.

Single-process executor only (the worker lives in the engine process).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm.sampling_params import SamplingType
from vllm.v1.core.sched.output import DecodeLoopSpec, SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.request import Request

logger = init_logger(__name__)

NSTOP = 8  # == worker decode_loop.NSTOP
_DEBUG = os.environ.get("VLLM_DECODE_LOOP_DEBUG") == "1"
MAX_ITERS = 8192  # == worker decode_loop.MAX_ITERS


MAX_TOPK = 64  # == sample/fused_topk.MAX_K


def _loop_kind(request: Request) -> str | None:
    """Loop kind of a request: greedy, topk (temperature + top-k <= 64 +
    top-p, the fused sampler graphed.py uses for such rows) or None (regular
    steps only)."""
    sp = request.sampling_params
    if (
        sp is None
        or request.pooling_params is not None
        or request.lora_request is not None
        or request.use_structured_output
        or request.has_encoder_inputs
        or request.kv_transfer_params is not None
    ):
        return None
    stop_ids = set(sp.stop_token_ids or ())
    if sp.eos_token_id is not None:
        stop_ids.add(sp.eos_token_id)
    plain = (
        sp.presence_penalty == 0.0
        and sp.frequency_penalty == 0.0
        and sp.repetition_penalty == 1.0
        and not sp.watermarking
        and sp.logprobs is None
        and sp.prompt_logprobs is None
        and sp.logprob_token_ids is None
        and sp.structured_outputs is None
        and not sp.logit_bias
        and sp.allowed_token_ids is None
        and not sp.bad_words
        and not sp.bad_words_token_ids
        and sp.extra_args is None
        and sp.thinking_token_budget is None
        and sp.trace_decode_token_ids is None
        and len(stop_ids) <= NSTOP
    )
    if not plain:
        return None
    if sp.sampling_type == SamplingType.GREEDY:
        return "greedy"
    if (
        sp.sampling_type == SamplingType.RANDOM  # no per-request seed
        and sp.temperature > 0.0
        and sp.min_p == 0.0
        and sp.min_tokens == 0
        and 0 < sp.top_k <= MAX_TOPK
    ):
        return "topk"
    return None


class _Active:
    def __init__(self, so: SchedulerOutput, req_ids: list[str]):
        self.so = so  # the scheduled first iteration
        self.req_ids = req_ids  # batch order (ring columns)
        self.first = True
        self.retired = [False] * len(req_ids)  # stopped by the GPU
        self.left = [so.decode_loop.budgets[r] for r in req_ids]
        self.stop_ids = [set(so.decode_loop.stop_ids[r]) for r in req_ids]


class DecodeLoopController:
    """Wraps EngineCore.step_fn."""

    def __init__(self, core: EngineCore, max_bs: dict[str, int]):
        self.core = core
        self.sched = core.scheduler
        self.inner = core.step_fn
        self.max_bs = max_bs
        self.max_n = max(max_bs.values())
        self._next_kind = "greedy"
        self.max_model_len = self.sched.max_model_len
        self.active: _Active | None = None
        self._kinds: dict[str, str | None] = {}
        # Running set the worker could not loop on (no matching graph): do not
        # retry until it changes.
        self._refused: frozenset[str] | None = None
        self.num_loops = 0
        self.num_loop_tokens = 0

    @classmethod
    def maybe_create(cls, core: EngineCore) -> DecodeLoopController | None:
        from vllm.v1.executor.uniproc_executor import UniProcExecutor

        if os.environ.get("VLLM_DECODE_LOOP", "1") == "0":
            return None
        if not isinstance(core.model_executor, UniProcExecutor):
            return None
        sched = core.scheduler
        if (
            getattr(sched, "num_spec_tokens", 0)
            or getattr(sched, "connector", None) is not None
            or getattr(sched, "ec_connector", None) is not None
            or core.vllm_config.cache_config.enable_prefix_caching
        ):
            return None
        try:
            max_bs = core.model_executor.collective_rpc(
                _worker_max_bs, single_value=True
            )
        except Exception:  # not the GPU model runner v2
            return None
        if not max_bs:
            return None
        logger.info("Device decode loop enabled (max requests per kind: %s)", max_bs)
        return cls(core, max_bs)

    # -- dispatch ---------------------------------------------------------------
    def step(self):
        if self.active is not None:
            return self._poll()
        if self._eligible():
            bq = self.core.batch_queue
            if bq:
                return self._drain_one()
            return self._start()
        return self.inner()

    def _kind(self, request: Request) -> str | None:
        rid = request.request_id
        if rid in self._kinds:
            return self._kinds[rid]
        if len(self._kinds) > 4096:
            self._kinds = {}
        k = self._kinds[rid] = _loop_kind(request)
        return k

    def _eligible(self) -> bool:
        why = self._why_not()
        if _DEBUG and why is not None:
            self._dbg_n = getattr(self, "_dbg_n", 0) + 1
            if self._dbg_n % 500 == 1:
                logger.info("decode loop not eligible: %s", why)
        return why is None

    def _why_not(self) -> str | None:
        s = self.sched
        running = s.running
        n = len(running)
        if n == 0 or n > self.max_n:
            return f"{n} running"
        if s.get_num_unfinished_requests() != n:
            return "waiting requests"
        if self._refused is not None:
            if self._refused == frozenset(r.request_id for r in running):
                return "refused"
            self._refused = None
        kind = None
        for req in running:
            if req.status != RequestStatus.RUNNING:
                return f"status {req.status}"
            # A decode: everything but the last sampled token is computed
            # (in-flight async steps hold placeholders).
            if (
                req.num_computed_tokens
                != req.num_tokens + req.num_output_placeholders - 1
            ):
                return "not a decode"
            if req.num_output_tokens < req.sampling_params.min_tokens:
                return "min_tokens"
            k = self._kind(req)
            if k is None or (kind is not None and k != kind):
                return "sampling params"
            kind = k
        if n > self.max_bs.get(kind, 0):
            return f"{n} {kind} requests"
        self._next_kind = kind
        return None

    def _drain_one(self):
        """Collect the oldest in-flight async step without scheduling a new
        one (the second half of EngineCore.step_with_batch_queue)."""
        core = self.core
        future, so, exec_fut = core.batch_queue.pop()
        with (
            core.capture_iteration_details(so) as details,
            core.log_error_detail(so),
        ):
            out = future.result()
            if out is None:
                exec_fut.result()
                raise RuntimeError("unexpected error")
        core._process_aborts_queue()
        outs = self.sched.update_from_output(so, out)
        core._attach_iteration_details(outs, details)
        core._attach_weight_version(outs)
        return outs, True

    # -- start ------------------------------------------------------------------
    def _start(self):
        core, s = self.core, self.sched
        so = s.schedule(core._should_throttle_prefills())
        reqs = [s.requests[r] for r in so.num_scheduled_tokens]
        if (
            so.scheduled_new_reqs
            or so.scheduled_spec_decode_tokens
            or len(reqs) != len(s.running)
            or any(v != 1 for v in so.num_scheduled_tokens.values())
        ):
            return self._run_regular(so)

        budgets: dict[str, int] = {}
        stop_ids: dict[str, list[int]] = {}
        kvm = s.kv_cache_manager
        cached = so.scheduled_cached_reqs
        pos_of = {rid: i for i, rid in enumerate(cached.req_ids)}
        for req in reqs:
            rid = req.request_id
            sp = req.sampling_params
            # check_stop after the j-th new token: num_tokens + j >= max_model_len
            # or num_output_tokens + j >= max_tokens.
            b = min(
                req.max_tokens - req.num_output_tokens,
                self.max_model_len - req.num_tokens,
                MAX_ITERS,
            )
            b = max(b, 1)
            if b > 1:
                # schedule() covered this step's token; add the rest.
                new = kvm.allocate_slots(req, b - 1)
                if new is None:
                    b = 1
                else:
                    extra = new.get_block_ids(allow_none=True)
                    i = pos_of.get(rid)
                    if extra is not None and any(extra) and i is not None:
                        cur = cached.new_block_ids[i]
                        cached.new_block_ids[i] = (
                            extra
                            if cur is None
                            else tuple(a + e for a, e in zip(cur, extra))
                        )
            budgets[rid] = b
            ids = set(sp.stop_token_ids or ())
            if sp.eos_token_id is not None:
                ids.add(sp.eos_token_id)
            stop_ids[rid] = sorted(ids)
        zero = kvm.take_new_block_ids()
        if zero and s.needs_kv_cache_zeroing:
            so.new_block_ids_to_zero = (so.new_block_ids_to_zero or []) + zero
        so.decode_loop = DecodeLoopSpec(
            budgets=budgets, stop_ids=stop_ids, kind=self._next_kind
        )

        ex = core.model_executor
        with core.log_error_detail(so):
            out = ex.execute_model(so)
        req_ids = ex.collective_rpc(_worker_loop_req_ids, single_value=True)
        if req_ids is None:
            # The worker ran a regular step (no matching loop graph).
            if _DEBUG:
                logger.info("decode loop refused by the worker")
            so.decode_loop = None
            self._refused = frozenset(so.num_scheduled_tokens)
            if out is None:
                out = ex.sample_tokens(s.get_grammar_bitmask(so))
            core._process_aborts_queue()
            outs = s.update_from_output(so, out)
            core._attach_weight_version(outs)
            return outs, True
        self.active = _Active(so, req_ids)
        self.num_loops += 1
        if _DEBUG:
            logger.info(
                "decode loop %d start (%s): %d reqs, budgets %s",
                self.num_loops,
                self._next_kind,
                len(req_ids),
                [budgets[r] for r in req_ids],
            )
            self._t0 = time.perf_counter()
            self._tok0 = self.num_loop_tokens
        return {}, True

    def _run_regular(self, so: SchedulerOutput):
        core = self.core
        ex = core.model_executor
        with core.log_error_detail(so):
            out = ex.execute_model(so)
            if out is None:
                out = ex.sample_tokens(self.sched.get_grammar_bitmask(so))
        core._process_aborts_queue()
        outs = self.sched.update_from_output(so, out)
        core._attach_weight_version(outs)
        return outs, so.total_num_scheduled_tokens > 0

    # -- poll -------------------------------------------------------------------
    def _poll(self):
        core, s, a = self.core, self.sched, self.active
        assert a is not None
        ex = core.model_executor
        rows, done = ex.collective_rpc(_worker_loop_poll, single_value=True)
        core._process_aborts_queue()
        merged: dict[int, EngineCoreOutputs] = {}
        if len(rows):
            self.num_loop_tokens += int((rows >= 0).sum())
            if a.first:
                a.first = False
                self._merge(merged, s.update_from_output(a.so, self._mro(a, rows[:1])))
                self._track(a, rows[:1])
                rows = rows[1:]
            if len(rows):
                self._merge(merged, self._synthetic_step(a, rows))
                self._track(a, rows)
        if done:
            ex.collective_rpc(_worker_loop_finish, single_value=True)
            self.active = None
            if _DEBUG:
                dt = time.perf_counter() - self._t0
                nt = self.num_loop_tokens - self._tok0
                logger.info(
                    "decode loop %d done: %d tokens in %.1f ms (%.3f ms/row-token)",
                    self.num_loops,
                    nt,
                    dt * 1e3,
                    dt * 1e3 / max(nt, 1) * len(a.req_ids),
                )
        elif self._should_stop(a):
            ex.collective_rpc(_worker_loop_stop, single_value=True)
        if not merged and not done:
            # Nothing new: give the GPU and the I/O threads some room.
            time.sleep(2e-5)
        core._attach_weight_version(merged)
        return merged, True

    def _mro(self, a: _Active, rows: np.ndarray) -> ModelRunnerOutput:
        req_ids, toks = [], []
        for j, rid in enumerate(a.req_ids):
            col = rows[:, j]
            t = col[col >= 0].tolist()
            if t:
                req_ids.append(rid)
                toks.append(t)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=toks,
        )

    def _synthetic_step(self, a: _Active, rows: np.ndarray):
        """Account several device iterations as one step whose requests each
        computed (and sampled) their new tokens, like a spec-decode step."""
        s = self.sched
        mro = self._mro(a, rows)
        so = SchedulerOutput.make_empty()
        total = 0
        for rid, toks in zip(mro.req_ids, mro.sampled_token_ids):
            req = s.requests.get(rid)
            if req is None or req.is_finished():
                continue
            k = len(toks)
            req.num_computed_tokens += k
            req.num_in_flight_tokens += k
            so.num_scheduled_tokens[rid] = k
            total += k
        so.total_num_scheduled_tokens = total
        if not total:
            return {}
        return s.update_from_output(so, mro)

    @staticmethod
    def _track(a: _Active, rows: np.ndarray) -> None:
        for j in range(len(a.req_ids)):
            if a.retired[j]:
                continue
            col = rows[:, j]
            t = col[col >= 0]
            if len(t) == 0:
                continue
            a.left[j] -= len(t)
            if a.left[j] <= 0 or int(t[-1]) in a.stop_ids[j] or (col == -2).any():
                a.retired[j] = True

    def _should_stop(self, a: _Active) -> bool:
        s = self.sched
        in_loop = sum(1 for r in a.req_ids if r in s.requests)
        if s.get_num_unfinished_requests() > in_loop:
            return True  # a new request: give it its prefill
        for j, rid in enumerate(a.req_ids):
            if a.retired[j]:
                continue
            req = s.requests.get(rid)
            if req is None or req.is_finished():
                return True  # finished on the host (abort, stop string, ...)
        is_running = getattr(self.core, "is_running", None)
        return is_running is not None and not is_running()

    @staticmethod
    def _merge(dst: dict[int, EngineCoreOutputs], src: dict[int, EngineCoreOutputs]):
        for ci, eco in src.items():
            cur = dst.get(ci)
            if cur is None:
                dst[ci] = eco
                continue
            cur.outputs.extend(eco.outputs)
            if eco.finished_requests:
                cur.finished_requests = (
                    cur.finished_requests or set()
                ) | eco.finished_requests
            if eco.scheduler_stats is not None:
                cur.scheduler_stats = eco.scheduler_stats


# Worker-side calls (run in the engine process by UniProcExecutor).
def _loop(worker):
    return worker.model_runner.decode_loop


def _worker_max_bs(worker) -> dict[str, int]:
    dl = getattr(worker.model_runner, "decode_loop", None)
    return dl.max_bs if dl is not None else {}


def _worker_loop_req_ids(worker):
    live = _loop(worker).live
    return None if live is None else live.req_ids


def _worker_loop_poll(worker):
    return _loop(worker).poll()


def _worker_loop_stop(worker):
    _loop(worker).stop()


def _worker_loop_finish(worker):
    _loop(worker).finish()
