# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side decode loop for a few greedy requests (worker side).

At one to eight decode requests the engine step is bound by host CPU time
(~2 ms of Python per step on slow hosts vs ~1.5 ms of GPU work), so the GPU
waits on the host between tokens. Here the whole decode loop runs on the GPU
instead: one CUDA graph with a while node whose body is the model forward,
the greedy argmax and a bookkeeping kernel (``_loop_tail_kernel``) that feeds
each token back as the next input, advances positions / seq_lens / slot
mappings, applies the same request-state updates as ``post_update``, retires
rows that reach a stop token or their token budget, and publishes the tokens
to a pinned host ring. The engine polls that ring and turns the rows into
ordinary outputs (vllm/v1/engine/decode_loop.py); it asks the GPU to stop
through a pinned flag when a new request arrives or a request is aborted.

The first iteration's inputs come from the regular path: the engine schedules
one normal decode step (with KV blocks preallocated for the whole budget) and
``execute_model`` prepares every input buffer as for a FULL graph replay, then
replays the loop graph instead. The loop graph is captured on the same
persistent buffers as the FULL decode graph of the same size, so it reads
exactly what that graph would read.

Retired rows keep running until the loop ends (the batch shape is static):
their Mamba-like state slots are pointed at the null block, their KV writes
at PAD_SLOT_ID and their seq_len at 0, like the padding rows of a regular
step, so they no longer touch the request's state. The loop
also ends as soon as the live rows fit a smaller captured size, so the
engine restarts it on the smaller (faster) graph.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

LOOP_SIZES = (1, 2, 4, 8)
MAX_ITERS = 8192  # ring rows; a loop runs at most this many iterations
NSTOP = 8  # stop token ids per row on the device
_MAXIT_DEBUG = int(os.environ.get("VLLM_DECODE_LOOP_MAXIT", str(MAX_ITERS)))

_GROUP_SKIP, _GROUP_MAMBA, _GROUP_ATTN = 0, 1, 2

# Backends whose decode metadata is all device-side views of the persistent
# input buffers (block tables, seq_lens, slot mappings, positions), so a graph
# replayed with those buffers advanced on the GPU stays valid. Backends that
# plan on the host (FlashInfer, FA3 scheduler metadata, ...) would replay
# stale plans after the first iteration.
_DEVICE_METADATA_BACKENDS = frozenset({"MAMBA3", "OLALA_DIFF_TPA", "OLALA_TPA_FACTOR"})


def enabled() -> bool:
    return os.environ.get("VLLM_DECODE_LOOP", "1") != "0"


class _HostPtr:
    """A pinned host tensor handed to Triton by its UVA address: kernels read
    and write host memory directly (zero-copy)."""

    def __init__(self, t: torch.Tensor):
        assert t.is_pinned()
        self.t, self.dtype = t, t.dtype

    def data_ptr(self) -> int:
        return self.t.data_ptr()


@triton.jit
def _loop_head_kernel(rem_ptr, maxit_ptr, stop_ptr, pred_ptr, BS: tl.constexpr):
    r = tl.arange(0, BS)
    live = tl.sum((tl.load(rem_ptr + r) > 0).to(tl.int32), axis=0)
    stop = tl.load(stop_ptr, volatile=True)
    tl.store(pred_ptr, (live > 0) & (stop == 0) & (tl.load(maxit_ptr) > 0))


@triton.jit
def _loop_tail_kernel(
    tok_ptr,  # [BS] int64, this iteration's argmax
    idx_ptr,  # [BS] int64, batch row -> request state index (-1: padding)
    rem_ptr,  # [BS] int32, tokens left per row (0: retired)
    stop_ids_ptr,  # [BS, NSTOP] int64, -1 padded
    input_ids_ptr,  # [BS] int32
    pos_ptr,  # [BS] int64
    seq_ptr,  # [BS] int32
    slot_map_ptr,  # [num_groups, max_tokens] int64
    slot_map_stride,
    grp_kind_ptr,  # [NG] int32
    grp_bsize_ptr,  # [NG] int32
    grp_bt_ptrs,  # [NG] uint64, gathered (input) block tables
    grp_bt_stride_ptr,  # [NG] int64
    num_computed_ptr,  # [max_reqs] int32
    last_sampled_ptr,  # [max_reqs] int64
    total_len_ptr,  # [max_reqs] int32
    all_tok_ptr,  # [max_reqs, max_model_len] int32
    all_tok_stride,
    bins_ptr,  # [max_reqs, vocab] int32 (output_bin_counts)
    bins_stride,
    it_ptr,
    maxit_ptr,
    stop_ptr,  # pinned host int32
    ring_ptr,  # pinned host int32 [MAX_ITERS * BS]
    pred_ptr,
    BS: tl.constexpr,
    NSTOP: tl.constexpr,
    NG: tl.constexpr,
    MIN_LIVE: tl.constexpr,
):
    r = tl.arange(0, BS)
    rem = tl.load(rem_ptr + r)
    active = rem > 0
    tok = tl.load(tok_ptr + r)
    req = tl.load(idx_ptr + r)
    it = tl.load(it_ptr)
    # Publish (-2: row not active this iteration).
    tl.store(ring_ptr + it * BS + r, tl.where(active, tok, -2).to(tl.int32))

    # Request-state updates of post_update (one sampled token, one computed).
    req_s = tl.where(active, req, 0)
    tl.store(last_sampled_ptr + req_s, tok, mask=active)
    tlen = tl.load(total_len_ptr + req_s, mask=active, other=0)
    tl.store(all_tok_ptr + req_s * all_tok_stride + tlen, tok.to(tl.int32), mask=active)
    tl.store(total_len_ptr + req_s, tlen + 1, mask=active)
    nc = tl.load(num_computed_ptr + req_s, mask=active, other=0)
    tl.store(num_computed_ptr + req_s, nc + 1, mask=active)
    bin_p = bins_ptr + req_s * bins_stride + tok
    cnt = tl.load(bin_p, mask=active, other=0)
    tl.store(bin_p, cnt + 1, mask=active)

    # Stop tokens and budget.
    j = tl.arange(0, NSTOP)
    sid = tl.load(stop_ids_ptr + r[:, None] * NSTOP + j[None, :])
    hit = tl.sum((sid == tok[:, None]).to(tl.int32), axis=1) > 0
    rem = tl.where(active, rem - 1, rem)
    rem = tl.where(active & hit, 0, rem)
    still = rem > 0
    retire = active & (rem == 0)

    # Next iteration's inputs.
    pos = tl.load(pos_ptr + r) + active.to(tl.int64)
    seq = tl.load(seq_ptr + r) + active.to(tl.int32)
    # A retired row looks like a padding row (seq_len 0): batch-wide values
    # such as the TPA split size see what the regular path would see.
    seq = tl.where(retire, 0, seq)
    tl.store(input_ids_ptr + r, tok.to(tl.int32), mask=active)
    tl.store(pos_ptr + r, pos)
    tl.store(seq_ptr + r, seq)
    tl.store(rem_ptr + r, rem)
    for g in tl.static_range(NG):
        kind = tl.load(grp_kind_ptr + g)
        bsz = tl.load(grp_bsize_ptr + g)
        bt = tl.cast(tl.load(grp_bt_ptrs + g), tl.pointer_type(tl.int32))
        bts = tl.load(grp_bt_stride_ptr + g)
        if kind == 2:
            blk = tl.load(bt + r * bts + pos // bsz, mask=still, other=0)
            slot = tl.where(still, blk.to(tl.int64) * bsz + pos % bsz, -1)
            tl.store(slot_map_ptr + g * slot_map_stride + r, slot, mask=active)
        if kind == 1:
            # Retired row: its recurrent state now lives in the null block.
            tl.store(bt + r * bts, tl.zeros_like(r), mask=retire)

    it += 1
    tl.store(it_ptr, it)
    live = tl.sum(still.to(tl.int32), axis=0)
    stop = tl.load(stop_ptr, volatile=True)
    tl.store(pred_ptr, (it < tl.load(maxit_ptr)) & (live >= MIN_LIVE) & (stop == 0))


class _LoopGraph:
    def __init__(self, bs: int, kind: str, device: torch.device):
        self.bs = bs
        self.kind = kind  # "greedy" (argmax) or "topk" (fused top-k / top-p sampler)
        # Per-row sampling parameters (kind "topk"): copied from the sampler's
        # tables at loop start, read by row index.
        self.rows = torch.arange(bs, dtype=torch.int64, device=device)
        self.temp = torch.ones(bs, dtype=torch.float32, device=device)
        self.top_k = torch.ones(bs, dtype=torch.int32, device=device)
        self.top_p = torch.ones(bs, dtype=torch.float32, device=device)
        self.seed = torch.zeros(bs, dtype=torch.int64, device=device)
        self.idx = torch.full((bs,), -1, dtype=torch.int64, device=device)
        self.rem = torch.zeros(bs, dtype=torch.int32, device=device)
        self.stop_ids = torch.full((bs, NSTOP), -1, dtype=torch.int64, device=device)
        self.it = torch.zeros(1, dtype=torch.int32, device=device)
        self.maxit = torch.zeros(1, dtype=torch.int32, device=device)
        self.pred = torch.zeros((), dtype=torch.bool, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None


class _Live:
    def __init__(self, L: _LoopGraph, req_ids: list[str], idx: np.ndarray, k: int):
        self.L = L
        self.req_ids = req_ids
        self.idx = idx
        self.k = k
        self.it = 0  # rows read so far
        self.produced = np.zeros(len(req_ids), dtype=np.int64)
        self.event = torch.cuda.Event()
        self.stop_sent = False


class DecodeLoop:
    def __init__(self, runner: GPUModelRunner):
        self.r = runner
        self.loops: dict[int, _LoopGraph] = {}
        self.live: _Live | None = None
        self.ring_np: np.ndarray | None = None
        self.stop_np: np.ndarray | None = None

    # -- capture --------------------------------------------------------------
    def _unsupported(self) -> str | None:
        r = self.r
        if not enabled():
            return "disabled (VLLM_DECODE_LOOP=0)"
        if r.cudagraph_manager is None:
            return "no CUDA graphs"
        if not r.cudagraph_manager.cudagraph_mode.has_full_cudagraphs():
            return "no FULL CUDA graphs"
        if r.speculator is not None or r.lora_config is not None:
            return "speculative decoding / LoRA"
        if r.pcp_manager is not None or r.dcp_size > 1 or r.dp_size > 1:
            return "context / data parallel"
        if not (r.is_first_pp_rank and r.is_last_pp_rank):
            return "pipeline parallel"
        if r.use_aux_hidden_state_outputs or r.uses_inputs_embeds:
            return "aux hidden states / inputs_embeds"
        if r.ubatch_runner is not None:
            return "micro-batching"
        if r.sampler is None:
            return "no sampler"
        if r.vllm_config.cache_config.enable_prefix_caching:
            return "prefix caching"
        if r.vllm_config.kv_transfer_config is not None:
            return "KV connector"
        if getattr(r.model_state, "recoverssm", None) is not None:
            return "recoverssm"
        if getattr(r.model_state, "_align_mode", False):
            return "mamba align mode"
        if r.model_config.rswa_window is not None:
            return "R-SWA"
        names = {g.backend.get_name() for groups in r.attn_groups for g in groups}
        if not names or not names <= _DEVICE_METADATA_BACKENDS:
            return f"attention backends {sorted(names)}"
        return None

    def _kinds(self) -> list[str]:
        """Greedy always; the fused top-k sampler only where the regular path
        samples with it too (graphed.py), so loop tokens equal regular ones."""
        from vllm.v1.worker.gpu.lowlat import LOWLAT
        from vllm.v1.worker.gpu.sample.sampler import Sampler

        s = self.r.sampler
        kinds = ["greedy"]
        if (
            type(s) is Sampler
            and LOWLAT.sampler_graph
            and LOWLAT.fused_topk
            and not s.compute_nans
            and not s.return_sampling_mask
            and s.trace_replay_state is None
            and not s.use_flashinfer
            and not s.use_xpu_sampler
        ):
            kinds.append("topk")
        return kinds

    def _groups(self):
        kinds, bsizes, bts, strides = [], [], [], []
        bt = self.r.block_tables
        for g, group in enumerate(self.r.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                kind = _GROUP_MAMBA
            elif isinstance(spec, AttentionSpec):
                kind = _GROUP_ATTN
            else:
                kind = _GROUP_SKIP
            if kind == _GROUP_ATTN and not bt._slot_mapping_enabled[g]:
                kind = _GROUP_SKIP
            kinds.append(kind)
            bsizes.append(bt.block_sizes[g])
            bts.append(bt.input_block_tables[g].data_ptr())
            strides.append(bt.input_block_tables[g].stride(0))
        dev = self.r.device
        self.grp_kind = torch.tensor(kinds, dtype=torch.int32, device=dev)
        self.grp_bsize = torch.tensor(bsizes, dtype=torch.int32, device=dev)
        self.grp_bt = torch.tensor(bts, dtype=torch.uint64, device=dev)
        self.grp_bt_stride = torch.tensor(strides, dtype=torch.int64, device=dev)
        self.ng = len(kinds)
        return kinds

    @torch.inference_mode()
    def capture(self) -> None:
        why = self._unsupported()
        if why is not None:
            logger.info("Device decode loop off: %s", why)
            return
        r = self.r
        captured = set(r.cudagraph_manager.captured_token_counts())
        sizes = [s for s in LOOP_SIZES if s <= r.max_num_reqs and s in captured]
        if not sizes:
            logger.info(
                "Device decode loop off: no FULL decode graph of size %s", LOOP_SIZES
            )
            return
        kinds = self._groups()
        if _GROUP_ATTN not in kinds and _GROUP_MAMBA not in kinds:
            logger.info("Device decode loop off: no KV cache groups")
            return
        ring = torch.full((MAX_ITERS * max(sizes),), -1, dtype=torch.int32).pin_memory()
        stop = torch.zeros(1, dtype=torch.int32).pin_memory()
        self.ring_np, self.stop_np = ring.numpy(), stop.numpy()
        self._ring, self._stop = _HostPtr(ring), _HostPtr(stop)
        self.pool = torch.cuda.graph_pool_handle()
        kinds = self._kinds()
        with graph_capture(device=r.device):
            for kind in kinds:
                for bs in sorted(sizes, reverse=True):
                    self.loops[(kind, bs)] = self._capture_one(bs, kind)
        torch.accelerator.synchronize()
        logger.info(
            "Device decode loop: captured batch sizes %s (%s)", sizes, ", ".join(kinds)
        )

    def _capture_one(self, bs: int, kind: str) -> _LoopGraph:
        from vllm.v1.worker.gpu.cudagraph_utils import prepare_inputs_to_capture
        from vllm.v1.worker.gpu.sample.fused_topk import MAX_K, fused_topk_topp_sample

        r = self.r
        L = _LoopGraph(bs, kind, r.device)
        ib = r.input_buffers

        def build():
            # Fresh metadata per run: attention-like layers may cache derived
            # device tensors (e.g. the TPA split size) on the metadata object,
            # and the captured graph must compute them itself.
            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                bs,
                bs,
                r.model_state,
                ib,
                r.block_tables,
                r.attn_groups,
                r.kv_cache_config,
                full_cudagraph=True,
                max_query_len=1,
            )
            ib.is_padding.fill_(True)
            return attn_metadata, slot_mappings

        model_inputs = {
            "input_ids": ib.input_ids[:bs],
            "positions": ib.positions[:bs],
            "intermediate_tensors": None,
            **r.model_state.prepare_dummy_inputs(bs, bs),
        }
        rs = r.req_states
        bins = r.sampler.penalties_state.output_bin_counts
        stream = torch.cuda.current_stream()

        def body(attn_metadata, slot_mappings):
            with set_forward_context(
                attn_metadata,
                r.vllm_config,
                num_tokens=bs,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=None,
                slot_mapping=slot_mappings,
                batch_descriptor=None,
                is_padding=ib.is_padding[:bs],
            ):
                hidden = r.model(**model_inputs)
            logits = r.model.compute_logits(hidden[:bs])
            if kind == "greedy":
                tok = logits.argmax(dim=-1)
            else:
                # graphed.py's fused path, on per-row tables
                tok = fused_topk_topp_sample(
                    logits,
                    L.rows,
                    L.temp,
                    L.top_k,
                    L.top_p,
                    L.seed,
                    ib.positions[:bs],
                    MAX_K,
                )
            _loop_tail_kernel[(1,)](
                tok,
                L.idx,
                L.rem,
                L.stop_ids,
                ib.input_ids,
                ib.positions,
                ib.seq_lens,
                r.block_tables.slot_mappings,
                r.block_tables.slot_mappings.stride(0),
                self.grp_kind,
                self.grp_bsize,
                self.grp_bt,
                self.grp_bt_stride,
                rs.num_computed_tokens.gpu,
                rs.last_sampled_tokens,
                rs.total_len.gpu,
                rs.all_token_ids.gpu,
                rs.all_token_ids.gpu.stride(0),
                bins,
                bins.stride(0),
                L.it,
                L.maxit,
                self._stop,
                self._ring,
                L.pred,
                BS=bs,
                NSTOP=NSTOP,
                NG=self.ng,
                MIN_LIVE=max(1, bs // 2 + 1) if bs > 1 else 1,
                num_warps=1,
            )

        # Warmup on dummy inputs with rem = 0: nothing is published and no
        # request state is touched (the forward writes the null block / PAD).
        L.rem.zero_()
        _loop_head_kernel[(1,)](L.rem, L.maxit, self._stop, L.pred, BS=bs, num_warps=1)
        body(*build())
        torch.accelerator.synchronize()
        g = torch.cuda.CUDAGraph()
        md = build()
        with torch.cuda.graph(g, pool=self.pool, stream=stream):
            _loop_head_kernel[(1,)](
                L.rem, L.maxit, self._stop, L.pred, BS=bs, num_warps=1
            )
            g.begin_capture_to_while_node(L.pred)
            body(*md)
            g.set_conditional_handle_for_current_node(L.pred)
            g.end_capture_to_conditional_node()
        L.graph = g
        return L

    # -- run ------------------------------------------------------------------
    @property
    def max_bs(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kind, bs in self.loops:
            out[kind] = max(out.get(kind, 0), bs)
        return out

    def can_start(
        self, input_batch: InputBatch, num_tokens_padded: int, cg_mode, spec
    ) -> bool:
        return (
            self.live is None
            and cg_mode == CUDAGraphMode.FULL
            and (spec.kind, num_tokens_padded) in self.loops
            and input_batch.num_tokens == input_batch.num_reqs
            and not input_batch.has_prefill
            and input_batch.num_draft_tokens == 0
        )

    @torch.inference_mode()
    def start(self, input_batch: InputBatch, num_tokens_padded: int, spec) -> None:
        """Replay the loop graph on the inputs execute_model just prepared.
        ``spec``: DecodeLoopSpec (budgets and stop ids by request id)."""
        L = self.loops[(spec.kind, num_tokens_padded)]
        bs, n = L.bs, input_batch.num_reqs
        req_ids = list(input_batch.req_ids)
        idx = np.full(bs, -1, dtype=np.int64)
        idx[:n] = input_batch.idx_mapping_np
        rem = np.zeros(bs, dtype=np.int32)
        stop_ids = np.full((bs, NSTOP), -1, dtype=np.int64)
        for i, rid in enumerate(req_ids):
            rem[i] = spec.budgets[rid]
            ids = spec.stop_ids[rid]
            stop_ids[i, : len(ids)] = ids
        k = int(min(rem.max(), MAX_ITERS, _MAXIT_DEBUG))
        if L.kind == "topk":
            st = self.r.sampler.sampling_states
            ridx = input_batch.idx_mapping_np
            for buf, tab, pad in (
                (L.temp, st.temperature, 1.0),
                (L.top_k, st.top_k, 1),
                (L.top_p, st.top_p, 1.0),
                (L.seed, st.seeds, 0),
            ):
                v = np.full(bs, pad, dtype=tab.np.dtype)
                v[:n] = tab.np[ridx]
                buf.copy_(torch.from_numpy(v), non_blocking=True)
        L.idx.copy_(torch.from_numpy(idx), non_blocking=True)
        L.rem.copy_(torch.from_numpy(rem), non_blocking=True)
        L.stop_ids.copy_(torch.from_numpy(stop_ids), non_blocking=True)
        L.it.zero_()
        L.maxit.fill_(k)
        assert self.ring_np is not None and self.stop_np is not None
        self.ring_np[: k * bs] = -1
        self.stop_np[0] = 0
        L.graph.replay()
        live = self.live = _Live(L, req_ids, idx[:n], k)
        live.event.record()

    def poll(self) -> tuple[np.ndarray, bool]:
        """Rows published since the last poll ([rows, num_reqs] int32, -2 for
        rows not active in that iteration) and whether the loop is over (all
        rows read)."""
        live = self.live
        assert live is not None and self.ring_np is not None
        done = live.event.query()
        bs, n = live.L.bs, len(live.req_ids)
        ring = self.ring_np[: live.k * bs].reshape(live.k, bs)
        it = live.it
        end = it
        while end < live.k and (ring[end, :n] != -1).all():
            end += 1
        rows = ring[it:end, :n].copy()
        live.it = end
        if rows.size:
            live.produced += (rows >= 0).sum(axis=0)
        # The graph is over: every iteration it ran has been published.
        return rows, done and (end == live.k or (ring[end, :n] == -1).all())

    def stop(self) -> None:
        if self.live is not None and not self.live.stop_sent:
            assert self.stop_np is not None
            self.stop_np[0] = 1
            self.live.stop_sent = True

    def finish(self) -> None:
        """After the loop graph completed: account the produced tokens in the
        CPU mirror of num_computed_tokens (the device state is up to date)."""
        live = self.live
        assert live is not None
        live.event.synchronize()
        self.r.req_states.num_computed_tokens_np[live.idx] += live.produced.astype(
            np.int32
        )
        self.live = None
