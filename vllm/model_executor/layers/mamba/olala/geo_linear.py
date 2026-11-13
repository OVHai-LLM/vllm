# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Geodesic residual update fused into the next projection (decode).

Every Olala block ends with ``h = geodesic(x, g)`` (``x`` the residual, ``g``
the mixer / MLP output) whose consumers are the next projections: the MoE
entry ``[w_in ; router]`` after the mixer, the Mamba ``in_proj`` after the
MLP. ``h = a * x + b * g`` with per-row scalars ``a, b`` from ``|x|², x·g,
|g|²``, so ``h @ W.T = a * (x @ W.T) + b * (g @ W.T)``: a short loop computes
the row reductions, the main loop streams ``W`` against the stacked operand
``[x ; g]`` and the epilogue combines the two halves. Each program also writes
one column slice of ``h`` (the next residual). The dot uses the exact
``x``/``g`` instead of the bf16-rounded ``h`` of the unfused path.

Opt-in (``OLALA_GEO_FUSE=1``): on B300 with HBM-resident weights the Triton
GEMM loses ~1.5 us to cuBLAS on these shapes, more than the saved launch.
"""

import os

import torch

from vllm.triton_utils import tl, triton

GEO_LINEAR_MAX_TOKENS = 32
GEO_FUSE = os.environ.get("OLALA_GEO_FUSE", "0") == "1"


@triton.jit
def _geo_linear_kernel(
    x_ptr,  # [M, K] residual
    g_ptr,  # [M, K] mixer / MLP output
    h_ptr,  # [M, K] out: geodesic(x, g)
    w_ptr,  # [N, K] bf16
    y_ptr,  # [M, N] out (bf16 or fp32)
    scale_ptr,
    bias_ptr,
    inv_depth,
    clamp_val,
    M,
    N,
    K,
    stride_x,
    stride_g,
    stride_h,
    stride_y,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    KC: tl.constexpr,
    HB: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.arange(0, BM)
    m_mask = m < M
    # Row reductions in their own loop: inside the dot loop they stop Triton
    # from pipelining the weight stream.
    xx = tl.zeros([BM], dtype=tl.float32)
    xg = tl.zeros([BM], dtype=tl.float32)
    gg = tl.zeros([BM], dtype=tl.float32)
    for c0 in range(0, K, KC):
        kc = c0 + tl.arange(0, KC)
        msk = m_mask[:, None] & (kc[None, :] < K)
        xf = tl.load(
            x_ptr + m[:, None] * stride_x + kc[None, :], mask=msk, other=0.0
        ).to(tl.float32)
        gf = tl.load(
            g_ptr + m[:, None] * stride_g + kc[None, :], mask=msk, other=0.0
        ).to(tl.float32)
        xx += tl.sum(xf * xf, axis=1)
        xg += tl.sum(xf * gf, axis=1)
        gg += tl.sum(gf * gf, axis=1)
    # Same math as olala._geodesic_norm_kernel.
    coeff = xg / tl.maximum(xx, 1e-12)
    tangent_sq = gg - 2.0 * coeff * xg + coeff * coeff * xx
    safe_tangent = tl.maximum(tl.sqrt(tl.maximum(tangent_sq, 0.0)), 1e-8)
    safe_r = tl.maximum(tl.sqrt(xx), 1e-6)
    scale = tl.load(scale_ptr).to(tl.float32)
    bias = tl.load(bias_ptr).to(tl.float32)
    theta = tl.minimum(safe_tangent / safe_r, clamp_val)
    theta = tl.minimum((theta * scale + bias) * inv_depth, clamp_val).to(tl.float32)
    kf = safe_r * tl.sin(theta) / safe_tangent
    cos_t = tl.cos(theta)
    a = cos_t - coeff * kf

    # One dot on the stacked operand [x ; g] (rows BM.. are g).
    r = tl.arange(0, 2 * BM)
    is_x = r < BM
    rr = tl.where(is_x, r, r - BM)
    row_ptr = tl.where(is_x, x_ptr + rr * stride_x, g_ptr + rr * stride_g)
    r_mask = rr < M
    offs_n = pid * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros([2 * BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        k_mask = offs_k < K
        op = tl.load(
            row_ptr[:, None] + offs_k[None, :],
            mask=r_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(op, tl.trans(w), acc)
    acc_x, acc_g = tl.split(tl.permute(tl.reshape(acc, (2, BM, BN)), (1, 2, 0)))
    y = a[:, None] * acc_x + kf[:, None] * acc_g
    tl.store(
        y_ptr + m[:, None] * stride_y + offs_n[None, :],
        y.to(y_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )
    if pid * HB < K:
        offs_h = pid * HB + tl.arange(0, HB)
        msk = m_mask[:, None] & (offs_h[None, :] < K)
        x = tl.load(
            x_ptr + m[:, None] * stride_x + offs_h[None, :], mask=msk, other=0.0
        ).to(tl.float32)
        g = tl.load(
            g_ptr + m[:, None] * stride_g + offs_h[None, :], mask=msk, other=0.0
        ).to(tl.float32)
        h = x * cos_t[:, None] + (g - coeff[:, None] * x) * kf[:, None]
        tl.store(
            h_ptr + m[:, None] * stride_h + offs_h[None, :],
            h.to(h_ptr.dtype.element_ty),
            mask=msk,
        )


def geo_linear(
    x: torch.Tensor,
    g: torch.Tensor,
    geo: torch.nn.Module,
    w: torch.Tensor,
    h_out: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Writes ``geo(x, g)`` into ``h_out`` and returns ``geo(x, g) @ w.T``.

    ``geo`` is an ``OlalaGeodesicNorm``; ``M <= GEO_LINEAR_MAX_TOKENS``."""
    M, K = x.shape
    N = w.shape[0]
    assert w.is_contiguous() and M <= GEO_LINEAR_MAX_TOKENS
    BN = 64 if N >= 4096 else 16
    BK = 256 if M <= 16 else 128
    BM = max(8, triton.next_power_of_2(M))
    grid = triton.cdiv(N, BN)
    y = torch.empty(M, N, dtype=out_dtype, device=x.device)
    _geo_linear_kernel[(grid,)](
        x, g, h_out, w, y, geo.scale, geo.bias,
        1.0 / (geo.layer_idx + 1), geo.clamp,
        M, N, K, x.stride(0), g.stride(0), h_out.stride(0), y.stride(0),
        BM=BM, BN=BN, BK=BK, KC=max(64, 4096 // BM),
        HB=triton.next_power_of_2(triton.cdiv(K, grid)),
        num_warps=4, num_stages=3,
    )
    return y
