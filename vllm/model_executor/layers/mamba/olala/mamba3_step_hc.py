"""Mamba-3 decode step for large batches with the preamble fused (see
csrc/mamba3_step_hc.cu), and the Q16 state format.

Takes the raw in_proj output ([in_proj | in_proj_dyn](u), bf16) and replaces
_decode_preamble_bc_kernel + the CUDA step. Each CTA owns one head and
prefetches the next token's state and rows while it updates the current one.
B300, graphed, one layer at B = 256 (preamble + step): fp32 state 142.5 us vs
156 us; Q16 state 116 us.

Q16: the fp32 SSM state stored as int16 mantissas (offset binary: q + 32768) plus one
fp32 scale per half row of 64 values (scale ~ max|h| / 32767, q = rint(h / scale)): 16.5
bits per value, relative error ~3e-5; teacher-forced decode NLL equal to the
fp32 state over 6 x 3000 tokens (bf16 rounding: +1.9%).
"""

from __future__ import annotations

from functools import lru_cache

import torch

from vllm.model_executor.layers.mamba.olala.pdl import pdl_enabled


@lru_cache(maxsize=1)
def _load():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    major, minor = torch.cuda.get_device_capability()
    sm = f"{major}{minor}"
    return load_extension(
        f"olala_mamba3_step_hc_sm{sm}",
        "mamba3_step_hc.cu",
        ["-O3", "-std=c++20", f"-gencode=arch=compute_{sm},code=sm_{sm}"],
    )


def supported(num_heads, head_dim, d_state, rank, num_angles, hidden) -> bool:
    return (num_heads, head_dim, d_state, rank, num_angles) == (48, 64, 128, 4, 32) and hidden == 1536


def mamba3_step_hc(zb, ssm_pool, k_pool, v_pool, angle_pool, slots, dt_bias, a_floor, wb, wc, eps,
                   Dp, xproj, zproj, outproj, bias_q, bias_k, y, ssm_scale=None, ctas_per_sm=0) -> None:
    """zb: (N, 6288 + 1056) bf16 in_proj output; wb / wc: fp32 B / C norm weights;
    xproj / zproj / outproj / bias_q / bias_k: fp32 (r, h, n) layouts; y: (N, 3072) bf16.
    ssm_pool fp32, or int16 (Q16) with ssm_scale (P, H, D, 2) fp32."""
    _load().step(zb, ssm_pool, ssm_scale, k_pool, v_pool, angle_pool, slots, dt_bias, a_floor, wb, wc, eps,
                 Dp, xproj, zproj, outproj, bias_q, bias_k, y, ctas_per_sm, pdl_enabled())


def q16_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 state (..., 128) -> (int16 offset-binary mantissas, fp32 scales (..., 2)). A valid encoding (|q| <= 32767,
    within one unit of the kernels' rounding); every encoder's output decodes the same way."""
    xs = x.view(*x.shape[:-1], 2, x.shape[-1] // 2)
    sc = xs.abs().amax(-1) / torch.full((), 32767.0, device=x.device)
    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
    q = torch.round(xs.double() * (1.0 / sc.double())[..., None]).clamp(-32767, 32767).to(torch.int32)
    return (q ^ 0x8000).to(torch.int16).view(x.shape), sc


def q16_dequantize(q: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    """h = (u - 32768) * scale, u the unsigned offset-binary mantissa."""
    qi = q.to(torch.int32) ^ -32768
    return (qi.view(*q.shape[:-1], 2, q.shape[-1] // 2).float() * sc[..., None]).view(q.shape)
