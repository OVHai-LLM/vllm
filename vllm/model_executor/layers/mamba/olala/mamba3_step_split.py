"""Mamba-3 decode step for 1-4 tokens with the preamble fused and the heads
split over CTAs (see csrc/mamba3_step_split.cu).

Takes the raw in_proj output ([in_proj | in_proj_dyn](u), bf16) and replaces
_decode_preamble_bc_kernel + the CUDA step: one launch instead of two, 384
CTAs (8 state rows each) instead of 48, and with PDL every pool / constant
load is issued under the in_proj GEMV. B300, graphed, one Mamba layer at
B = 1 (GEMV included): 10.6 us vs 13.5 us.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from vllm.model_executor.layers.mamba.olala.pdl import pdl_enabled

MAX_TOKENS = 4


@lru_cache(maxsize=1)
def _load():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    major, minor = torch.cuda.get_device_capability()
    sm = f"{major}{minor}"
    return load_extension(
        f"olala_mamba3_step_split_sm{sm}",
        "mamba3_step_split.cu",
        ["-O3", "-std=c++20", f"-gencode=arch=compute_{sm},code=sm_{sm}"],
    )


@lru_cache(maxsize=8)
def _counters(device: torch.device) -> torch.Tensor:
    # per-head reader counts; every launch leaves them at zero (launches on one
    # stream never overlap past griddepcontrol.wait, so one buffer serves all layers)
    return torch.zeros(48, dtype=torch.int32, device=device)


def supported(
    num_heads, head_dim, d_state, rank, num_angles, ssm_dtype, hidden
) -> bool:
    return (
        (num_heads, head_dim, d_state, rank, num_angles) == (48, 64, 128, 4, 32)
        and ssm_dtype == torch.float32
        and hidden == 1536
    )


def mamba3_step_split(
    zb,
    ssm_pool,
    k_pool,
    v_pool,
    angle_pool,
    slots,
    dt_bias,
    a_floor,
    wb,
    wc,
    eps,
    Dp,
    xproj,
    zproj,
    outproj,
    bias_q,
    bias_k,
    y,
) -> None:
    """zb: (N, 6288 + 1056) bf16 in_proj output, N <= 4; wb / wc: fp32 B / C
    norm weights; xproj / zproj / outproj / bias_q / bias_k: fp32 (r, h, n)
    layouts; y: (N, 3072) bf16."""
    _load().step(
        zb,
        _counters(zb.device),
        ssm_pool,
        k_pool,
        v_pool,
        angle_pool,
        slots,
        dt_bias,
        a_floor,
        wb,
        wc,
        eps,
        Dp,
        xproj,
        zproj,
        outproj,
        bias_q,
        bias_k,
        y,
        pdl_enabled(),
    )
