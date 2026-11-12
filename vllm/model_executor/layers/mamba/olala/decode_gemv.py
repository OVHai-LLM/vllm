# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-token GEMV for Olala's decode projections.

cuBLAS's batch-1 kernels sit on a ~4.2 us floor whatever the matrix size,
so the many small projections of a decode step (latent MoE, in_proj_dyn,
TPA factors) run at well under 1 TB/s. This streaming kernel — each program
owns BN output rows and walks K in BK chunks — is 2.5-3 us on those shapes
and also ahead of cuBLAS on the large ones (measured under CUDA graphs on
GH200). For M > 1 the per-row loop re-reads the weight tile, so cuBLAS
remains the better choice and ``decode_gemv`` falls back to F.linear.
"""

import os
from functools import cache

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


_DISABLED = os.environ.get("OLALA_GEMV", "1") == "0"


@triton.jit
def _gemv_m1_kernel(
    x_ptr, w_ptr, y_ptr, N, K, stride_wn,
    BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        x = tl.load(x_ptr + offs_k)
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def _gemv_m1_full_kernel(
    x_ptr, w_ptr, y_ptr, N, K, stride_wn,
    BN: tl.constexpr, BK: tl.constexpr,
):
    """Whole (padded) K extent per program: every load is in flight at once
    instead of walking K in a dependent loop."""
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    n_mask = offs_n < N
    k_mask = offs_k < K
    w = tl.load(
        w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
        mask=n_mask[:, None] & k_mask[None, :], other=0.0,
    )
    x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0)
    acc = tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


@cache
def is_blackwell() -> bool:
    return torch.cuda.get_device_capability()[0] >= 10


def _decode_gemv_blackwell(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """B300 (weights streamed from HBM, CUDA graphs): the full-K kernel is
    2.7 us on the latent MoE projections where the K loop takes 4-12 us, and
    3.8 / 5.1 us on mixer_proj (K = 3072 / 4608) vs 4.5 / 9.0 us. Wide outputs
    (N >= 4096: in_proj, the TPA projections) are faster on cuBLAS."""
    N, K = weight.shape
    bk = triton.next_power_of_2(K)
    if N >= 4096 or bk > 8192:
        return F.linear(x, weight)
    if bk <= 2048:
        bn, nw = 2, 8
    elif bk <= 4096:
        bn, nw = 1, 4
    else:
        bn, nw = 2, 2
    y = torch.empty(1, N, dtype=x.dtype, device=x.device)
    _gemv_m1_full_kernel[(triton.cdiv(N, bn),)](
        x, weight, y, N, K, weight.stride(0), BN=bn, BK=bk, num_warps=nw,
    )
    return y


def decode_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for ``x: (M, K)``, ``weight: (N, K)``; triton at M == 1."""
    if x.shape[0] != 1 or not x.is_cuda or x.dtype != weight.dtype or _DISABLED:
        return F.linear(x, weight)
    if is_blackwell():
        return _decode_gemv_blackwell(x, weight)
    N, K = weight.shape
    if K % 128 != 0:
        return F.linear(x, weight)
    y = torch.empty(1, N, dtype=x.dtype, device=x.device)
    # HBM-streaming configs measured on GH200 (36 distinct weights, graphed):
    # wide outputs like 8 rows/program; narrow outputs want more programs.
    if N >= 4096:
        bn, bk, nw = 8, 512, 4
    elif K % 1024 == 0 and K >= 3072:
        bn, bk, nw = 4, 1024, 8
    else:
        bn, bk, nw = 4, (512 if K % 512 == 0 else 128), 8
    _gemv_m1_kernel[(triton.cdiv(N, bn),)](
        x, weight, y, N, K, weight.stride(0), BN=bn, BK=bk, num_warps=nw,
    )
    return y


def _olala_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return decode_gemv(x, weight)


def _olala_linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)


# Opaque to torch.compile so the M == 1 dispatch is re-evaluated per cudagraph
# capture instead of being specialized by the first (large) compile.
direct_register_custom_op(
    op_name="olala_linear",
    op_func=_olala_linear,
    mutates_args=[],
    fake_impl=_olala_linear_fake,
)


def olala_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.olala_linear(x, weight)


def cat_rows_shared(params: list[torch.nn.Parameter], pad_to: int = 1) -> torch.Tensor:
    """Row-concatenate ``params`` and rebind each one to its slice of the result.

    The concatenated decode weight then costs no extra memory (a separate copy
    is ~0.8 GB over the 36 layers, a fifth of the KV budget on a 20 GB GPU).
    Row slices of a contiguous tensor keep the parameters' shapes and strides,
    and in-place weight reloads (``param.data.copy_``) land in the shared
    storage, so the concatenation never goes stale.

    With ``pad_to`` > 1 the buffer gets zero rows up to a multiple of ``pad_to`` (the returned
    tensor includes them): cuBLAS picks Hopper kernels only for aligned output widths.
    """
    with torch.no_grad():
        n_rows = sum(p.shape[0] for p in params)
        n_pad = -(-n_rows // pad_to) * pad_to
        ref = params[0].data
        w = torch.zeros(n_pad, *ref.shape[1:], dtype=ref.dtype, device=ref.device)
        torch.cat([p.data for p in params], 0, out=w[:n_rows])
        off = 0
        for p in params:
            n = p.shape[0]
            p.data = w[off : off + n]
            off += n
    return w
