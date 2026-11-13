# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala medium-batch latent MoE (grouped GEMMs) must match the small-batch
gather-GEMV path, which computes the same math token by token, including when
the routing piles many tokens on a few experts."""

import pytest
import torch

from vllm.model_executor.layers.mamba.olala.latent_moe_decode import olala_moe_small
from vllm.model_executor.layers.mamba.olala.latent_moe_medium import olala_moe_medium


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("m", [9, 64, 256, 512])
@pytest.mark.parametrize("skew", [0.0, 4.0])
def test_medium_path_matches_small_path(m: int, skew: float):
    torch.manual_seed(m)
    E, K, H, D1, N1, S = 256, 6, 1536, 384, 768, 1536
    dev = "cuda"
    hidden = torch.randn(m, H, device=dev).bfloat16()
    # skew = 4: a handful of experts take most of the batch.
    logits = torch.randn(m, E, device=dev) + skew * torch.randn(E, device=dev)
    bias = torch.randn(E, device=dev) * 0.1
    w_in = (torch.randn(D1 + S, H, device=dev) * H**-0.5).bfloat16()
    w_out = (torch.randn(H, D1 + S, device=dev) * (D1 + S) ** -0.5).bfloat16()
    w13 = (torch.randn(E, N1, D1, device=dev) * D1**-0.5).bfloat16()
    w2 = (torch.randn(E, D1, N1, device=dev) * N1**-0.5).bfloat16()

    out = olala_moe_medium(hidden, logits, bias, w_in, w_out, w13, w2, K, 2.45)
    ref = olala_moe_small(hidden, logits, bias, w_in, w_out, w13, w2, K, 2.45)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("m", [1, 4, 16, 64])
def test_fp32_entry_gemm_matches_bf16_entry(m: int):
    """A precomputed fp32 x_in (router merged into the input GEMM) is rounded
    to bf16 on load: same result as the bf16 input GEMM, on both paths."""
    from vllm.model_executor.layers.mamba.olala.latent_moe_decode import (
        SMALL_BATCH_MAX_TOKENS_WITH_MEDIUM,
    )

    torch.manual_seed(m)
    E, K, H, D1, N1, S = 256, 6, 1536, 384, 768, 1536
    dev = "cuda"
    hidden = torch.randn(m, H, device=dev).bfloat16()
    bias = torch.randn(E, device=dev) * 0.1
    w_in = (torch.randn(D1 + S, H, device=dev) * H**-0.5).bfloat16()
    gate = (torch.randn(E, H, device=dev) * H**-0.5).bfloat16()
    w_out = (torch.randn(H, D1 + S, device=dev) * (D1 + S) ** -0.5).bfloat16()
    w13 = (torch.randn(E, N1, D1, device=dev) * D1**-0.5).bfloat16()
    w2 = (torch.randn(E, D1, N1, device=dev) * N1**-0.5).bfloat16()

    y = torch.mm(hidden, torch.cat([w_in, gate]).T, out_dtype=torch.float32)
    x_in, logits = y[:, : D1 + S], y[:, D1 + S :]
    small = m <= SMALL_BATCH_MAX_TOKENS_WITH_MEDIUM
    fn = olala_moe_small if small else olala_moe_medium
    ref = fn(hidden, logits.contiguous(), bias, w_in, w_out, w13, w2, K, 2.45)
    out = fn(hidden, logits, bias, w_in, w_out, w13, w2, K, 2.45, x_in=x_in)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nan_padded_rows_do_not_corrupt_real_rows():
    """CUDA-graph padding rows may carry NaN logits; they must not change the
    real rows (a NaN row picking one expert k times overflowed its list)."""
    torch.manual_seed(0)
    E, K, H, D1, N1, S = 256, 6, 1536, 384, 768, 1536
    dev = "cuda"
    m, pad = 12, 4
    hidden = torch.randn(m + pad, H, device=dev).bfloat16()
    logits = torch.randn(m + pad, E, device=dev)
    logits[m:] = float("nan")
    bias = torch.randn(E, device=dev) * 0.1
    w_in = (torch.randn(D1 + S, H, device=dev) * H**-0.5).bfloat16()
    w_out = (torch.randn(H, D1 + S, device=dev) * (D1 + S) ** -0.5).bfloat16()
    w13 = (torch.randn(E, N1, D1, device=dev) * D1**-0.5).bfloat16()
    w2 = (torch.randn(E, D1, N1, device=dev) * N1**-0.5).bfloat16()
    out = olala_moe_medium(hidden, logits, bias, w_in, w_out, w13, w2, K, 2.45)
    ref = olala_moe_medium(
        hidden[:m], logits[:m], bias, w_in, w_out, w13, w2, K, 2.45
    )
    torch.testing.assert_close(out[:m].float(), ref.float(), atol=2e-2, rtol=2e-2)
