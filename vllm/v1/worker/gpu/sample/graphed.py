# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graphed logits + sampling for small pure-decode batches.

At one or two requests the decode step is bound by host CPU time, and the
eager ``compute_logits`` + ``Sampler`` path is ~10 launches and ~0.5 ms of
Python per step. For the common case (random sampling with temperature /
top-k / top-p, no logprobs, penalties, logit bias, bad words, thinking
budget, grammar or drafts) the exact same kernels are replayed from a CUDA
graph instead. Every other batch takes the eager path.

The graph reads the model's static output buffer, the static positions
buffer and the sampling-state UVA tables in place, so graphs are keyed on
those addresses (the UVA tables alternate between two slots when new
requests stage writes). Only ``expanded_idx_mapping`` is copied in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu.lowlat import LOWLAT
from vllm.v1.worker.gpu.sample.fused_topk import MAX_K, fused_topk_topp_sample
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler, apply_top_k_top_p

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

MAX_GRAPHED_REQS = 32
_MAX_GRAPHS = 128


class GraphedSampler:
    def __init__(self, model: torch.nn.Module, sampler: Sampler, device: torch.device):
        self.model = model
        self.sampler = sampler
        self.device = device
        self.graphs: dict[tuple, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        self.pool = None
        self.ones: dict[int, torch.Tensor] = {}
        self.zeros: dict[int, torch.Tensor] = {}
        self.disabled = type(sampler) is not Sampler

    def eligible(self, input_batch: InputBatch, grammar_output) -> bool:
        s = self.sampler
        n = input_batch.num_reqs
        return (
            not self.disabled
            and grammar_output is None
            and 0 < n <= MAX_GRAPHED_REQS
            and input_batch.num_tokens == n
            and input_batch.num_draft_tokens == 0
            and not input_batch.has_prefill
            and not s.compute_nans
            and not s.return_sampling_mask
            and s.trace_replay_state is None
            and not s.use_flashinfer
            and not s.use_xpu_sampler
            and bool(s.plain_sampling[input_batch.idx_mapping_np].all())
        )

    def __call__(self, hidden_states: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        s = self.sampler
        st = s.sampling_states
        n = input_batch.num_reqs
        idx_np = input_batch.idx_mapping_np
        needs_proc = bool(s.needs_logits_processing[idx_np].any())
        temp_np = st.temperature.np[idx_np]
        do_temp = needs_proc and not bool(np.all(temp_np == 1.0))
        top_k_np = st.top_k.np[idx_np]
        top_p_np = st.top_p.np[idx_np]
        do_top_k = bool(np.any(top_k_np != st.vocab_size))
        do_top_p = bool(np.any(top_p_np != 1.0))
        topp_only = bool(do_top_p and np.any((top_k_np >= st.vocab_size) & (top_p_np < 1.0)))
        # Every row with an active top-k <= MAX_K: one fused two-kernel pass (fused_topk.py)
        fused = bool(LOWLAT.fused_topk and do_top_k and np.all(top_k_np <= MAX_K))
        hs = hidden_states[:n]
        pos = input_batch.positions[:n]
        tabs = (st.temperature.gpu, st.seeds.gpu, st.top_k.gpu, st.top_p.gpu)
        key = (n, hs.data_ptr(), pos.data_ptr(), *(t.data_ptr() for t in tabs),
               needs_proc, do_temp, do_top_k, do_top_p, topp_only, fused)
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(key, hs, pos, tabs, n, needs_proc, do_temp, do_top_k, do_top_p, topp_only, fused)
        graph, idx_buf, out = entry
        idx_buf.copy_(input_batch.expanded_idx_mapping[:n], non_blocking=True)
        graph.replay()
        if n not in self.ones:
            self.ones[n] = torch.ones(n, dtype=torch.int32, device=self.device)
            self.zeros[n] = torch.zeros(n, dtype=torch.int32, device=self.device)
        return SamplerOutput(
            # A fresh tensor: the next replay overwrites ``out`` while this
            # step's token is still being copied to the host.
            sampled_token_ids=out.clone().view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=self.ones[n],
            num_rejected=self.zeros[n],
            sampling_mask_tensors=None,
        )

    def _body(self, hs, pos, idx, tabs, needs_proc, do_temp, do_top_k, do_top_p, topp_only, fused):
        temperature, seeds, top_k_tab, top_p_tab = tabs
        logits = self.model.compute_logits(hs)
        if fused:
            return fused_topk_topp_sample(
                logits, idx, temperature, top_k_tab, top_p_tab, seeds, pos, MAX_K
            )
        if needs_proc:
            logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
            if do_temp:
                apply_temperature(logits, idx, temperature)
        top_k = top_k_tab[idx] if do_top_k else None
        top_p = top_p_tab[idx] if do_top_p else None
        logits = apply_top_k_top_p(logits, top_k, top_p, has_topp_only_rows=topp_only)
        return gumbel_sample(
            logits, idx, temperature, seeds, pos,
            apply_temperature=False, is_drafting=False, use_fp64=self.sampler.use_fp64_gumbel,
        )

    def _capture(self, key, hs, pos, tabs, n, *flags):
        if len(self.graphs) >= _MAX_GRAPHS:
            self.graphs.clear()
        idx_buf = torch.zeros(n, dtype=torch.int64, device=self.device)
        # Eager warmup: builds Triton kernels and the top-k/top-p buffer caches outside capture.
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._body(hs, pos, idx_buf, tabs, *flags)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self._body(hs, pos, idx_buf, tabs, *flags)
        if self.pool is None:
            self.pool = graph.pool()
        logger.debug("graphed sampler: captured %s", key)
        self.graphs[key] = (graph, idx_buf, out)
        return self.graphs[key]

