# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala Mamba3 Q16 SSM state (OLALA_SSM_Q16): int16 offset-binary mantissas
plus one fp32 scale per half row of 64 values.

* prefill: the pooled hybrid prefill on Q16 pools gives bitwise the outputs of
  fp32 pools holding the dequantized states (same state-pass kernel);
* decode: mamba3_step_hc / mamba3_step_split on Q16 pools track the fp32
  kernels on the dequantized states, stay finite when a head forgets its
  state entirely (alpha underflow), and leave padded rows (slot -1) alone.
"""

import sys
from pathlib import Path

import pytest
import torch

from vllm.platforms import current_platform

H, D, S, R, NA = 48, 64, 128, 4, 32
N1 = H * (2 * D + 3)
ZB = N1 + 2 * R * S + NA
EPS, A_FLOOR = 1e-6, 1e-4

requires_sm90 = pytest.mark.skipif(
    not current_platform.is_cuda() or current_platform.get_device_capability().major < 9,
    reason="TMA step kernels need sm_90+",
)


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm()).item()


def _weights(gen):
    f = dict(device="cuda", dtype=torch.float32)
    rnd = lambda *s, sc=1.0: torch.randn(*s, generator=gen, **f) * sc  # noqa: E731
    return dict(
        dt_bias=rnd(H, sc=0.5) - 3.0, wb=rnd(S, sc=0.1), wc=rnd(S, sc=0.1), Dp=rnd(H),
        xpj=rnd(R, H, D, sc=0.3) + 0.25, zpj=rnd(R, H, D, sc=0.3) + 1.0, opj=rnd(R, H, D, sc=0.3) + 0.25,
        bias_q=rnd(R, H, S, sc=0.3) + 1.0, bias_k=rnd(R, H, S, sc=0.3) + 1.0,
    )


def _zb(n, gen, full_decay_rows=()):
    zb = torch.randn(n, ZB, generator=gen, device="cuda")
    zx = zb[:, :N1].view(n, H, 2 * D + 3)
    zx[..., 2 * D] = zx[..., 2 * D] * 1.5  # dt
    for b in full_decay_rows:  # dt * |A| ~ 400: alpha = exp(A dt) underflows to 0
        zx[b, :, 2 * D] = 20.0
        zx[b, :, 2 * D + 1] = 20.0
    return zb.bfloat16()


def _paged(shapes, P, pad=4096):
    """State tensors laid out like vLLM's hybrid KV cache: views into one buffer
    with a slot stride of one (padded) page."""
    sizes = [torch.Size(s).numel() * torch.empty((), dtype=d).element_size() for s, d in shapes]
    page = sum(sizes) + pad
    raw = torch.zeros(P * page, dtype=torch.uint8, device="cuda")
    out, off = [], 0
    for (s, d), nb in zip(shapes, sizes):
        es = torch.empty((), dtype=d).element_size()
        out.append(torch.as_strided(raw[off:].view(d), (P, *s), (page // es, *torch.empty(s).stride())))
        off += nb
    return out


def _pools(P, q16, gen):
    shapes = [((H, D, S), torch.int16 if q16 else torch.float32), ((R, H, S), torch.bfloat16),
              ((H, D), torch.bfloat16), ((H, NA), torch.float32)]
    if q16:
        shapes.append(((H, D, 2), torch.float32))
    return _paged(shapes, P)


def _step(kind, W, zb, pools, slots):
    from vllm.model_executor.layers.mamba.olala.mamba3_step_hc import mamba3_step_hc
    from vllm.model_executor.layers.mamba.olala.mamba3_step_split import mamba3_step_split

    y = torch.empty(zb.shape[0], H * D, dtype=torch.bfloat16, device="cuda")
    ssm, kp, vp, ang = pools[:4]
    fn = mamba3_step_split if kind == "split" else mamba3_step_hc
    fn(zb, ssm, kp, vp, ang, slots, W["dt_bias"], A_FLOOR, W["wb"], W["wc"], EPS, W["Dp"], W["xpj"], W["zpj"],
       W["opj"], W["bias_q"], W["bias_k"], y, ssm_scale=pools[4] if len(pools) > 4 else None)
    return y


@requires_sm90
@pytest.mark.parametrize("kind,nb", [("hc", 2), ("hc", 7), ("hc", 64), ("split", 1), ("split", 2)])
@torch.inference_mode()
def test_q16_step_tracks_fp32(kind, nb):
    from vllm.model_executor.layers.mamba.olala.mamba3_step_hc import q16_dequantize, q16_quantize

    gen = torch.Generator(device="cuda").manual_seed(nb)
    W = _weights(gen)
    P = nb + 4
    slots = (torch.randperm(P - 1, generator=gen, device="cuda")[:nb] + 1).to(torch.int32)
    if nb > 4:
        slots[3] = -1  # padded row: no writes, y = 0
    h0 = torch.randn(P, H, D, S, generator=gen, device="cuda") * 0.5
    q0, s0 = q16_quantize(h0)
    rest = [(torch.randn(P, R, H, S, generator=gen, device="cuda") * 0.3).bfloat16(),
            (torch.randn(P, H, D, generator=gen, device="cuda") * 0.3).bfloat16(),
            torch.rand(P, H, NA, generator=gen, device="cuda") * 6.0]
    pf, pq = _pools(P, False, gen), _pools(P, True, gen)
    pf[0].copy_(q16_dequantize(q0, s0))
    pq[0].copy_(q0)
    pq[4].copy_(s0)
    for i, t in enumerate(rest, 1):
        pf[i].copy_(t)
        pq[i].copy_(t)
    valid = slots >= 0
    for step in range(16):
        zb = _zb(nb, gen, full_decay_rows=(0,) if step == 5 else ())
        yf, yq = _step(kind, W, zb, pf, slots), _step(kind, W, zb, pq, slots)
        assert torch.isfinite(yq.float()).all() and torch.isfinite(pq[4]).all()
        assert (yq[~valid] == 0).all()
        sl = slots[valid].long()
        assert _rel(yq[valid], yf[valid]) < 5e-3, step
        assert _rel(q16_dequantize(pq[0], pq[4])[sl], pf[0][sl]) < 3e-4, step
    untouched = torch.ones(P, dtype=torch.bool, device="cuda")
    untouched[slots[valid].long()] = False
    assert torch.equal(pq[0][untouched], q0[untouched]) and torch.equal(pq[4][untouched], s0[untouched])


@requires_sm90
@torch.inference_mode()
def test_fp32_hc_equals_split():
    """fp32 state: the large-batch kernel is bitwise the split kernel."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    W = _weights(gen)
    P = 6
    slots = torch.tensor([4, 1], dtype=torch.int32, device="cuda")
    init = [torch.randn(P, H, D, S, generator=gen, device="cuda") * 0.5,
            (torch.randn(P, R, H, S, generator=gen, device="cuda") * 0.3).bfloat16(),
            (torch.randn(P, H, D, generator=gen, device="cuda") * 0.3).bfloat16(),
            torch.rand(P, H, NA, generator=gen, device="cuda") * 6.0]
    a, b = _pools(P, False, gen), _pools(P, False, gen)
    for x, y, t in zip(a, b, init):
        x.copy_(t)
        y.copy_(t)
    for _ in range(8):
        zb = _zb(2, gen)
        assert torch.equal(_step("hc", W, zb, a, slots), _step("split", W, zb, b, slots))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.skipif(
    not current_platform.is_cuda() or current_platform.get_device_capability().major != 10,
    reason="tcgen05 prefill kernels are sm_100 / sm_103 only",
)
@pytest.mark.parametrize("lens,init,ang_scale", [
    ([300, 650, 77], [False] * 3, 30.0),
    ([300, 2100, 77], [True, False, True], 30.0),
    ([9000, 1000], [False, False], 1.0),
])
@torch.inference_mode()
def test_q16_prefill_matches_fp32(lens, init, ang_scale):
    sys.path.insert(0, str(Path(__file__).parent))
    import test_mamba3_mimo as M

    from vllm.model_executor.layers.mamba.olala.mamba3_step_hc import q16_dequantize, q16_quantize
    from vllm.model_executor.layers.mamba.ops.mamba3 import mimo_ssd

    args = M._hybrid_inputs(sum(lens))
    args[4] = args[4] * ang_scale  # frequent 2 pi wraps
    ns = len(lens)
    g = torch.Generator(device="cuda").manual_seed(7)
    pools = [torch.randn(ns, M.H, M.A, device="cuda", generator=g),
             torch.randn(ns, M.H, M.P, M.N, device="cuda", generator=g) * 0.1,
             torch.randn(ns, M.R, M.H, M.N, device="cuda", generator=g).bfloat16(),
             torch.randn(ns, M.H, M.P, device="cuda", generator=g).bfloat16()]
    q, sc = q16_quantize(pools[1])
    p32 = [x.clone() for x in pools]
    p32[1] = q16_dequantize(q, sc)
    pq = [x.clone() for x in pools] + [sc.clone()]
    pq[1] = q.clone()
    has_init = torch.tensor(init, device="cuda") if any(init) else None
    cu = [0] + torch.tensor(lens).cumsum(0).tolist()
    slots = torch.arange(ns, device="cuda", dtype=torch.int32)
    o32 = torch.empty(sum(lens), M.H, M.P, device="cuda", dtype=torch.bfloat16)
    oq = torch.empty_like(o32)
    mimo_ssd.mamba3_mimo_hybrid_pools(*args, cu, p32, slots, has_init, o32)
    mimo_ssd.mamba3_mimo_hybrid_pools(*args, cu, pq, slots, has_init, oq)
    assert torch.equal(o32, oq)
    for i in (0, 2, 3):
        assert torch.equal(p32[i], pq[i])
    assert _rel(q16_dequantize(pq[1], pq[4]), p32[1]) < 5e-5
    assert ((pq[1].int() ^ -32768).abs() <= 32767).all()  # |q| <= 32767
