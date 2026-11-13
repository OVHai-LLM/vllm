# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala latent MoE, medium batches: up GEMM, relu^2 and down GEMM in one
kernel per (expert row block, intermediate slice).

Same routing and compaction as latent_moe_medium. A program owns BLOCK_M
rows of one expert and an ISPLIT-th of the intermediate dimension; for each
BI-wide block of it, ``h = relu^2(A @ W1[blk].T)`` stays on chip and is
consumed at once by ``acc += h @ W2[:, blk].T``. The intermediate never goes
to HBM; each slice writes its weighted fp32 partial ``[D2]`` per entry and
the epilogue sums them over (slice, k) in a fixed order (deterministic), then
applies the routed scale and concatenates relu^2 of the shared pre-act.

Opt-in (``OLALA_MOE_UPDOWN=1``).
"""

import os

import torch

from vllm.model_executor.layers.mamba.olala.latent_moe_medium import (
    _compact_kernel,
    _route_topk_kernel,
)
from vllm.model_executor.layers.mamba.olala.pdl import pdl_kwargs
from vllm.triton_utils import tl, triton

UPDOWN_ENABLED = os.environ.get("OLALA_MOE_UPDOWN", "0") == "1"


@triton.jit
def _moe_updown_kernel(
    a_ptr,  # latent [M, D1] (fp32 or bf16, rounded to bf16 on load)
    w1_ptr,  # [E, N1, D1] bf16
    w2_ptr,  # [E, D2, N1] bf16
    part_ptr,  # [ISPLIT, M*TOPK, D2] fp32 out
    tw_ptr,  # [M*TOPK] fp32
    rows_ptr,  # [E, RCAP] int32
    cnt_ptr,  # [E] int32
    MK,
    RCAP,
    stride_am,
    N1: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D2P: tl.constexpr,  # D2 padded to a power of 2
    E: tl.constexpr,
    TOPK: tl.constexpr,
    ISPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BI: tl.constexpr,
    BK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid_m = tl.program_id(0)
    split = tl.program_id(1)
    offs_e = tl.arange(0, E)
    nblk = (tl.load(cnt_ptr + offs_e) + BLOCK_M - 1) // BLOCK_M
    cum = tl.cumsum(nblk, axis=0)
    e = tl.sum((cum <= pid_m).to(tl.int32), axis=0)
    if e >= E:
        return
    first = tl.sum(tl.where(offs_e == e, cum - nblk, 0), axis=0)
    e = e.to(tl.int64)
    cnt = tl.load(cnt_ptr + e)
    offs_r = (pid_m - first) * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = offs_r < cnt
    entry = tl.load(rows_ptr + e * RCAP + offs_r, mask=rmask, other=0).to(tl.int64)
    arow = entry // TOPK
    offs_k = tl.arange(0, BK)
    offs_i = tl.arange(0, BI)
    offs_d2 = tl.arange(0, D2P)
    d2_mask = offs_d2 < D2
    w1_base = w1_ptr + e * N1 * D1
    w2_base = w2_ptr + e * D2 * N1
    acc = tl.zeros((BLOCK_M, D2P), dtype=tl.float32)
    i_lo = split * (N1 // ISPLIT)
    for i0 in range(i_lo, i_lo + N1 // ISPLIT, BI):
        hacc = tl.zeros((BLOCK_M, BI), dtype=tl.float32)
        for k0 in range(0, D1, BK):
            a = tl.load(
                a_ptr + arow[:, None] * stride_am + (k0 + offs_k)[None, :],
                mask=rmask[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            w1 = tl.load(
                w1_base + (i0 + offs_i)[None, :] * D1 + (k0 + offs_k)[:, None]
            )
            hacc = tl.dot(a, w1, hacc)
        h = hacc.to(tl.bfloat16).to(tl.float32)
        h = tl.maximum(h, 0.0)
        h = (h * h).to(tl.bfloat16)
        w2 = tl.load(
            w2_base + offs_d2[None, :] * N1 + (i0 + offs_i)[:, None],
            mask=d2_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(h, w2, acc)
    w = tl.load(tw_ptr + entry, mask=rmask, other=0.0)
    tl.store(
        part_ptr + split * MK * D2 + entry[:, None] * D2 + offs_d2[None, :],
        acc * w[:, None],
        mask=rmask[:, None] & d2_mask[None, :],
    )


@triton.jit
def _updown_epilogue_kernel(
    part_ptr,  # [ISPLIT, M*K, D2] fp32
    s_ptr,  # [M, S] shared-expert pre-activation (fp32 or bf16)
    cat_ptr,  # [M, D2 + S] bf16 out: [scale * sum part | relu2(s)]
    scale,
    S_LD,
    MK,
    D2: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    K: tl.constexpr,
    ISPLIT: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
    m = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BS + tl.arange(0, BS)
    if nb * BS < D2:
        acc = tl.zeros([BS], dtype=tl.float32)
        for sp in range(ISPLIT):
            for k in range(K):
                acc += tl.load(part_ptr + (sp * MK + m * K + k) * D2 + offs)
        tl.store(cat_ptr + m * (D2 + S) + offs, (acc * scale).to(tl.bfloat16))
    else:
        so = offs - D2
        v = tl.load(s_ptr + m * S_LD + so).to(tl.bfloat16).to(tl.float32)
        v = tl.maximum(v, 0.0)
        tl.store(cat_ptr + m * (D2 + S) + offs, (v * v).to(tl.bfloat16))


def _config(M: int) -> tuple[int, int, int, int, int]:
    """(ISPLIT, BLOCK_M, BI, BK, num_warps), tuned on B300 with HBM-resident
    experts (uniform and skewed routing)."""
    if M <= 64:
        return 8, 16, 32, 128, 4
    return 2, 16, 128, 128, 4


def olala_moe_updown(
    hidden: torch.Tensor,  # [M, hidden] bf16
    router_logits: torch.Tensor,  # [M, E] fp32 (any row stride)
    e_score_bias: torch.Tensor,  # [E] fp32
    w_in: torch.Tensor,  # [latent + shared_inter, hidden]
    w_out: torch.Tensor,  # [hidden, latent + shared_inter]
    w13: torch.Tensor,  # [E, inter, latent]
    w2: torch.Tensor,  # [E, latent, inter]
    top_k: int,
    routed_scale: float,
    x_in: torch.Tensor | None = None,
    config: tuple[int, int, int, int, int] | None = None,
) -> torch.Tensor:
    """shared(x) + fc2(scale * sum_k w_k * E_k(fc1(x))), routed experts in one
    fused up/down kernel."""
    M = hidden.shape[0]
    dev = hidden.device
    E, N1, D1 = w13.shape
    D2 = w2.shape[1]
    S = w_in.shape[0] - D1
    if x_in is None:
        x_in = torch.nn.functional.linear(hidden, w_in)
    MK = M * top_k
    isplit, bm, bi, bk, nw = config or _config(M)
    assert N1 % (isplit * bi) == 0 and D1 % bk == 0

    ids = torch.empty(MK, dtype=torch.int32, device=dev)
    tw = torch.empty(MK, dtype=torch.float32, device=dev)
    rows = torch.empty(E, M, dtype=torch.int32, device=dev)
    cnt = torch.empty(E, dtype=torch.int32, device=dev)
    RB = 8
    _route_topk_kernel[(triton.cdiv(M, RB),)](
        router_logits, e_score_bias, ids, tw, M, router_logits.stride(0),
        E=E, K=top_k, KP=triton.next_power_of_2(top_k), RB=RB, num_warps=4,
        **pdl_kwargs(),
    )
    _compact_kernel[(E,)](
        ids, rows, cnt, MK, M, QP=max(triton.next_power_of_2(MK), 16), num_warps=4,
        **pdl_kwargs(),
    )
    part = torch.empty(isplit, MK, D2, dtype=torch.float32, device=dev)
    _moe_updown_kernel[(E + triton.cdiv(MK, bm), isplit)](
        x_in, w13, w2, part, tw, rows, cnt, MK, M, x_in.stride(0),
        N1=N1, D1=D1, D2=D2, D2P=triton.next_power_of_2(D2), E=E, TOPK=top_k, ISPLIT=isplit,
        BLOCK_M=bm, BI=bi, BK=bk, num_warps=nw, **pdl_kwargs(),
    )
    cat = torch.empty(M, D2 + S, dtype=hidden.dtype, device=dev)
    _updown_epilogue_kernel[(M, (D2 + S) // 128)](
        part, x_in[:, D1:], cat, routed_scale, x_in.stride(0), MK,
        D2=D2, S=S, BS=128, K=top_k, ISPLIT=isplit, num_warps=1, **pdl_kwargs(),
    )
    return torch.nn.functional.linear(cat, w_out)
