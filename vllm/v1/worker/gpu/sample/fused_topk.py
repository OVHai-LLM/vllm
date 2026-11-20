# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused temperature + top-k + top-p + Gumbel-max sampling for a few rows.

For small batches with an active top-k (k <= 64) the eager pipeline
(fp32 copy, temperature, Qrita top-k/top-p over the full vocabulary with one
program per row, then the Gumbel argmax) is ~200 us of GPU per step on B300,
dominated by single-CTA vocabulary sweeps. Here:

  1. ``_cand_kernel`` (rows x chunks): each program divides its vocabulary
     chunk by the temperature and keeps its local top-``KC`` (value, index).
  2. ``_select_kernel`` (rows): merges the candidates, keeps the k largest
     (equal values ordered by lower token id), applies top-p on the softmax
     of the kept set (ascending cumulative mass <= 1 - p is dropped, the
     largest token always stays) and returns the Gumbel-max token with the
     sampler's own noise (murmur3 of seed, position, token).

The top-k set is exactly ``apply_top_k_top_p_triton``'s (the eager path's:
k tokens, ties at the k-th value resolved to the lower token ids; the
PyTorch reference keeps all ties instead). Top-p matches it up to float
rounding at the boundary (Qrita searches the threshold iteratively). The
temperature division and the Gumbel noise are bitwise those of
``apply_temperature`` and ``gumbel_sample``.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import _log1p_neg_stable, murmur3_uniform32

MAX_K = 64
_CHUNK = 4096


@triton.jit
def _okey(v, idx, V: tl.constexpr):
    """uint64 key ordered like (value desc, index asc)."""
    b = v.to(tl.int32, bitcast=True)
    ob = tl.where(b < 0, ~b, b | (-2147483648))
    return (ob.to(tl.uint32).to(tl.uint64) << 32) | (V - 1 - idx).to(tl.uint64)


@triton.jit
def _kval(key):
    ob = (key >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
    b = tl.where(ob < 0, ob & 2147483647, ~ob)
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _cand_kernel(
    logits_ptr, logits_stride, idx_ptr, temp_ptr, cand_ptr,
    V: tl.constexpr, CHUNK: tl.constexpr, NB: tl.constexpr, KC: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    req = tl.load(idx_ptr + row)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    offs = cb * CHUNK + tl.arange(0, CHUNK)
    m = offs < V
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=m, other=0.0).to(tl.float32)
    x = x / temp
    x = tl.where(m, x, float("-inf"))
    key = tl.where(m, _okey(x, offs, V), 0)
    top = tl.topk(key, KC)
    tl.store(cand_ptr + (row * NB + cb) * KC + tl.arange(0, KC), top)


@triton.jit
def _select_kernel(
    cand_ptr, idx_ptr, topk_ptr, topp_ptr, seeds_ptr, pos_ptr, out_ptr,
    V: tl.constexpr, NC: tl.constexpr, NCP: tl.constexpr, KS: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(idx_ptr + row)
    k = tl.load(topk_ptr + req)
    p = tl.load(topp_ptr + req).to(tl.float32)
    seed = tl.load(seeds_ptr + req)
    pos = tl.load(pos_ptr + row)
    o = tl.arange(0, NCP)
    keys = tl.load(cand_ptr + row * NC + o, mask=o < NC, other=0)
    top = tl.topk(keys, KS)  # descending (value, then lower index)
    r = tl.arange(0, KS)
    vals = _kval(top)
    tok = (V - 1 - (top & 0xFFFFFFFF)).to(tl.int32)
    kept = (r < k) & (top != 0)  # k largest, ties -> lower token id
    # top-p on the softmax of the kept set; drop ascending mass <= 1 - p
    mx = tl.max(tl.where(kept, vals, float("-inf")), axis=0)
    e = tl.where(kept, tl.exp(vals - mx), 0.0)
    pr = e / tl.sum(e, axis=0)
    asc = tl.sum(pr, axis=0) - tl.cumsum(pr, axis=0) + pr  # mass of this and all smaller
    drop = (asc <= 1.0 - p) & (r != 0)
    kept = kept & ~drop
    u = murmur3_uniform32(seed, pos, tok)
    g = -tl.log(-_log1p_neg_stable(u))
    score = tl.where(kept, vals + g, float("-inf"))
    _, best = tl.max(score, axis=0, return_indices=True)
    tl.store(out_ptr + row, tl.sum(tl.where(r == best, tok, 0), axis=0).to(tl.int64))


def fused_topk_topp_sample(
    logits: torch.Tensor,           # [n, V]
    idx: torch.Tensor,              # [n] request slots
    temperature: torch.Tensor,      # [max_reqs] fp32
    top_k: torch.Tensor,            # [max_reqs] int32
    top_p: torch.Tensor,            # [max_reqs] fp32
    seeds: torch.Tensor,            # [max_reqs] int64
    pos: torch.Tensor,              # [n]
    kmax: int,
) -> torch.Tensor:
    n, V = logits.shape
    KC = max(16, triton.next_power_of_2(kmax))
    KS = KC
    NB = triton.cdiv(V, _CHUNK)
    NC = NB * KC
    cand = torch.empty(n, NC, dtype=torch.uint64, device=logits.device)
    _cand_kernel[(n, NB)](logits, logits.stride(0), idx, temperature, cand,
                          V=V, CHUNK=_CHUNK, NB=NB, KC=KC, num_warps=8)
    out = torch.empty(n, dtype=torch.int64, device=logits.device)
    _select_kernel[(n,)](cand, idx, top_k, top_p, seeds, pos, out,
                         V=V, NC=NC, NCP=triton.next_power_of_2(NC), KS=KS, num_warps=8)
    return out
