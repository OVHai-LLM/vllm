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
