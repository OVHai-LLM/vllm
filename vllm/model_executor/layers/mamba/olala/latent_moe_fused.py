# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala latent MoE: routing, compaction, up GEMM, relu^2 and down GEMM in ONE
persistent kernel.

Phases (one launch, grid = resident programs only, so the grid barrier is safe):

1. prologue: each program routes a slice of the tokens (sigmoid, bias top-k,
   renormalized weights — same semantics as the small/medium paths) and appends
   every (token, k) entry to its expert's row list with an atomic slot counter;
2. grid barrier (atomic arrival counter, acquire spin);
3. tile loop: a tile is (expert, BM rows, intermediate slice). For each BI-wide
   block of the slice: h = relu^2(A @ W1[blk]^T) stays in registers/shared
   memory and is immediately consumed by acc += h @ W2[:, blk]^T. The
   intermediate never touches HBM. The tile writes its weighted fp32 partial
   [BM, D2] for its slice;
4. the last program to finish re-zeroes the counters for the next call.

The epilogue kernel sums the partials over (slice, k) in a fixed order
(deterministic), applies the routed scale and concatenates relu^2 of the
shared-expert pre-activation, like the small/medium paths.

The intermediate slice split (ISPLIT) trades parallelism (tiles) against the
partial traffic; rows within an expert list are in atomic (arbitrary) order,
which does not change any row's result.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _tile(
    t,
    i0,
    i1,
    cnt,
    nblk,
    cum,
    a_ptr,
    w1_ptr,
    w2_ptr,
    part_ptr,
    tw_ptr,
    rows_ptr,
    stride_am,
    RCAP,
    MK,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    D1: tl.constexpr,
    N1: tl.constexpr,
    D2: tl.constexpr,
    BM: tl.constexpr,
    BI: tl.constexpr,
    ISPLIT: tl.constexpr,
):
    """Blocks [i0, i1) of tile t: h = relu^2(A W1^T) kept on chip, acc += h W2^T,
    weighted partial written (whole tile) or atomically added (shared tile)."""
    IS: tl.constexpr = N1 // ISPLIT
    NI: tl.constexpr = IS // BI
    DC: tl.constexpr = 128  # D1 == D2 == 3 * DC
    offs_e = tl.arange(0, E)
    offs_c = tl.arange(0, DC)
    mb = t // ISPLIT
    s = t % ISPLIT
    e = tl.sum((cum <= mb).to(tl.int32), axis=0)
    first = tl.sum(tl.where(offs_e == e, cum - nblk, 0), axis=0)
    cnt_e = tl.sum(tl.where(offs_e == e, cnt, 0), axis=0)
    offs_r = (mb - first) * BM + tl.arange(0, BM)
    rmask = offs_r < cnt_e
    ent = tl.load(
        rows_ptr + e * RCAP + offs_r, mask=rmask, other=0, cache_modifier=".cg"
    ).to(tl.int64)
    a_base = a_ptr + (ent // TOPK)[:, None] * stride_am + offs_c[None, :]
    am = rmask[:, None]
    a0 = tl.load(a_base, mask=am, other=0.0).to(tl.bfloat16)
    a1 = tl.load(a_base + DC, mask=am, other=0.0).to(tl.bfloat16)
    a2 = tl.load(a_base + 2 * DC, mask=am, other=0.0).to(tl.bfloat16)
    e64 = e.to(tl.int64)
    ii = s * IS + tl.arange(0, BI)
    # W1[e, i, d] as [d, i] tiles; W2[e, n, i] as [i, n] tiles
    w1_base = w1_ptr + e64 * N1 * D1 + offs_c[:, None] + ii[None, :] * D1
    w2_base = w2_ptr + e64 * D2 * N1 + ii[:, None] + offs_c[None, :] * N1
    acc0 = tl.zeros([BM, DC], dtype=tl.float32)
    acc1 = tl.zeros([BM, DC], dtype=tl.float32)
    acc2 = tl.zeros([BM, DC], dtype=tl.float32)
    for i in range(i0 * BI, i1 * BI, BI):
        hp = tl.dot(a0, tl.load(w1_base + i * D1))
        hp = tl.dot(a1, tl.load(w1_base + i * D1 + DC), hp)
        hp = tl.dot(a2, tl.load(w1_base + i * D1 + 2 * DC), hp)
        h = tl.maximum(hp.to(tl.bfloat16).to(tl.float32), 0.0)
        h = (h * h).to(tl.bfloat16)
        acc0 = tl.dot(h, tl.load(w2_base + i), acc0)
        acc1 = tl.dot(h, tl.load(w2_base + i + DC * N1), acc1)
        acc2 = tl.dot(h, tl.load(w2_base + i + 2 * DC * N1), acc2)
    w = tl.load(tw_ptr + ent, mask=rmask, other=0.0, cache_modifier=".cg")[:, None]
    p = part_ptr + s * MK * D2 + ent[:, None] * D2 + offs_c[None, :]
    if (i0 == 0) & (i1 == NI):
        tl.store(p, acc0 * w, mask=am)
        tl.store(p + DC, acc1 * w, mask=am)
        tl.store(p + 2 * DC, acc2 * w, mask=am)
    else:
        tl.atomic_add(p, acc0 * w, mask=am, sem="relaxed")
        tl.atomic_add(p + DC, acc1 * w, mask=am, sem="relaxed")
        tl.atomic_add(p + 2 * DC, acc2 * w, mask=am, sem="relaxed")


@triton.jit
def _fused_moe_kernel(
    logits_ptr,  # [M, E] fp32
    bias_ptr,  # [E] fp32
    a_ptr,  # [M, D1] bf16 or fp32 latent (rounded to bf16), row stride stride_am
    w1_ptr,  # [E, N1, D1] bf16
    w2_ptr,  # [E, D2, N1] bf16
    part_ptr,  # [ISPLIT, M*TOPK, D2] fp32 out, zero at entry
    tw_ptr,  # [M*TOPK] fp32 scratch: routing weights
    rows_ptr,  # [E, RCAP] int32 scratch: entries of each expert
    cnt_ptr,  # [E] int32, zero at entry (re-zeroed at exit)
    sync_ptr,  # [2] int32, zero at entry (re-zeroed at exit)
    M,
    stride_l,
    stride_am,
    RCAP,
    NPROG,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    KP: tl.constexpr,
    RB: tl.constexpr,
    D1: tl.constexpr,
    N1: tl.constexpr,
    D2: tl.constexpr,
    BM: tl.constexpr,
    BI: tl.constexpr,
    ISPLIT: tl.constexpr,
    SK: tl.constexpr,
):
    pid = tl.program_id(0)
    MK = M * TOPK

    # ---- 1. routing + compaction ------------------------------------------
    offs_e = tl.arange(0, E)
    offs_k = tl.arange(0, KP)
    for rb in range(pid, tl.cdiv(M, RB), NPROG):
        rows = rb * RB + tl.arange(0, RB)
        rmask = rows < M
        lg = tl.load(
            logits_ptr + rows[:, None] * stride_l + offs_e[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        # NaN logits (padded rows under CUDA graphs) -> score 0: the top-k then
        # ranks by bias alone and still picks K distinct experts, so no expert
        # list can overflow its RCAP = M slots.
        sc = tl.where(lg != lg, 0.0, 1.0 / (1.0 + tl.exp(-lg)))
        sb = sc + tl.load(bias_ptr + offs_e)[None, :]
        sb = tl.where(sb != sb, float("-inf"), sb)
        kid = tl.zeros([RB, KP], dtype=tl.int32)
        kw = tl.zeros([RB, KP], dtype=tl.float32)
        for kk in tl.static_range(TOPK):
            mx = tl.max(sb, axis=1)
            idx = tl.minimum(
                tl.min(tl.where(sb == mx[:, None], offs_e[None, :], E), axis=1), E - 1
            )
            hit = offs_e[None, :] == idx[:, None]
            w = tl.sum(tl.where(hit, sc, 0.0), axis=1)
            sel = offs_k[None, :] == kk
            kid = tl.where(sel, idx[:, None], kid)
            kw = tl.where(sel, w[:, None], kw)
            sb = tl.where(hit, float("-inf"), sb)
        kw = kw / tl.maximum(tl.sum(kw, axis=1), 1e-20)[:, None]
        m2 = rmask[:, None] & (offs_k[None, :] < TOPK)
        entry = rows[:, None] * TOPK + offs_k[None, :]
        tl.store(tw_ptr + entry, kw, mask=m2)
        slot = tl.atomic_add(cnt_ptr + kid, 1, mask=m2, sem="relaxed")
        tl.store(rows_ptr + kid * RCAP + slot, entry, mask=m2)

    # ---- 2. grid barrier --------------------------------------------------
    tl.debug_barrier()
    tl.atomic_add(sync_ptr, 1, sem="release")
    arrived = tl.atomic_add(sync_ptr, 0, sem="acquire")
    while arrived < NPROG:
        arrived = tl.atomic_add(sync_ptr, 0, sem="acquire")
    tl.debug_barrier()

    # ---- 3. tiles: (expert, row block, intermediate slice) ----------------
    cnt = tl.load(cnt_ptr + offs_e, cache_modifier=".cg")
    nblk = (cnt + BM - 1) // BM
    cum = tl.cumsum(nblk, axis=0)
    ntiles = tl.sum(nblk, axis=0) * ISPLIT
    NI: tl.constexpr = N1 // ISPLIT // BI
    if SK:
        # Stream-K over (tile, BI block) steps: equal work per program. L >= NI,
        # so a tile is shared by at most two programs (atomic add on the zeroed
        # buffer). The split point of a row depends on the atomic slot order of
        # the prologue, so low bits are not run-to-run reproducible.
        total = ntiles * NI
        L = tl.maximum(tl.cdiv(total, NPROG), NI)
        j = pid * L
        end = tl.minimum(j + L, total)
        while j < end:
            t = j // NI
            i0 = j - t * NI
            i1 = tl.minimum(NI, i0 + (end - j))
            _tile(
                t,
                i0,
                i1,
                cnt,
                nblk,
                cum,
                a_ptr,
                w1_ptr,
                w2_ptr,
                part_ptr,
                tw_ptr,
                rows_ptr,
                stride_am,
                RCAP,
                MK,
                E,
                TOPK,
                D1,
                N1,
                D2,
                BM,
                BI,
                ISPLIT,
            )
            j += i1 - i0
    else:
        for t in range(pid, ntiles, NPROG):
            _tile(
                t,
                0,
                NI,
                cnt,
                nblk,
                cum,
                a_ptr,
                w1_ptr,
                w2_ptr,
                part_ptr,
                tw_ptr,
                rows_ptr,
                stride_am,
                RCAP,
                MK,
                E,
                TOPK,
                D1,
                N1,
                D2,
                BM,
                BI,
                ISPLIT,
            )

    # ---- 4. last program out resets the counters --------------------------
    tl.debug_barrier()
    done = tl.atomic_add(sync_ptr + 1, 1, sem="acq_rel")
    if done == NPROG - 1:
        tl.store(cnt_ptr + offs_e, tl.zeros([E], dtype=tl.int32))
        tl.atomic_xchg(sync_ptr, 0)
        tl.atomic_xchg(sync_ptr + 1, 0)


@triton.jit
def _fused_epilogue_kernel(
    part_ptr,  # [ISPLIT, M*K, D2] fp32, zeroed after reading
    s_ptr,  # [M, S] bf16/fp32 shared pre-activation (row stride S_LD)
    cat_ptr,  # [M, D2 + S] bf16 out: [scale * sum part | relu2(s)]
    scale,
    S_LD,
    MK,
    D2: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    K: tl.constexpr,
    ISPLIT: tl.constexpr,
):
    m = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BS + tl.arange(0, BS)
    if nb * BS < D2:
        acc = tl.zeros([BS], dtype=tl.float32)
        for k in tl.static_range(K):
            for s in tl.static_range(ISPLIT):
                pp = part_ptr + (s * MK + m * K + k) * D2 + offs
                acc += tl.load(pp)
                tl.store(pp, tl.zeros([BS], dtype=tl.float32))
        tl.store(cat_ptr + m * (D2 + S) + offs, (acc * scale).to(tl.bfloat16))
    else:
        so = offs - D2
        v = tl.load(s_ptr + m * S_LD + so).to(tl.bfloat16).to(tl.float32)
        v = tl.maximum(v, 0.0)
        tl.store(cat_ptr + m * (D2 + S) + offs, (v * v).to(tl.bfloat16))


# (max tokens, BM, BI, ISPLIT, num_warps, num_stages, programs per SM, stream-K)
_CONFIGS = [
    (64, 16, 32, 3, 4, 4, 1, False),
    (512, 16, 32, 2, 4, 4, 1, False),
]

_scratch: dict = {}


def _counters(dev: torch.device, E: int):
    key = (dev, E)
    buf = _scratch.get(key)
    if buf is None:
        buf = torch.zeros(E + 2, dtype=torch.int32, device=dev)
        _scratch[key] = buf
    return buf[:E], buf[E:]


def _zeroed_partials(dev: torch.device, n: int) -> torch.Tensor:
    """Persistent fp32 partial buffer, kept zero between calls by the epilogue."""
    key = ("part", dev)
    buf = _scratch.get(key)
    if buf is None or buf.numel() < n:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("fused MoE partial buffer must be sized before capture")
        buf = torch.zeros(max(n, 3 * 512 * 6 * 384), dtype=torch.float32, device=dev)
        _scratch[key] = buf
    return buf[:n]


def _resident_per_sm(kernel, dev: torch.device) -> int:
    """Programs of ``kernel`` that fit on one SM at once (the grid barrier
    deadlocks if the grid is larger than what is resident)."""
    key = ("occ", kernel.hash, dev)
    if key not in _scratch:
        kernel._init_handles()  # fills n_regs
        p = torch.cuda.get_device_properties(dev)
        threads = 32 * kernel.metadata.num_warps
        regs = ((max(kernel.n_regs, 1) * 32 + 255) // 256 * 256) * (threads // 32)
        smem = kernel.metadata.shared + 1024
        _scratch[key] = (
            p.multi_processor_count,
            min(
                p.regs_per_multiprocessor // regs,
                p.shared_memory_per_multiprocessor // smem,
                p.max_threads_per_multi_processor // threads,
            ),
        )
    return _scratch[key]


def _pick(M: int):
    for c in _CONFIGS:
        if c[0] >= M:
            return c[1:]
    return _CONFIGS[-1][1:]


def moe_routed_fused(
    latent: torch.Tensor,  # [M, D1] bf16 (may be a strided view)
    router_logits: torch.Tensor,  # [M, E] fp32
    e_score_bias: torch.Tensor,  # [E] fp32
    w13: torch.Tensor,  # [E, N1, D1] bf16
    w2: torch.Tensor,  # [E, D2, N1] bf16
    top_k: int,
    config=None,
):
    """Routing + up + relu^2 + down in one launch. Returns the weighted fp32
    partials [ISPLIT, M*K, D2] (routed scale not applied) and ISPLIT."""
    M, D1 = latent.shape
    E, N1, _ = w13.shape
    D2 = w2.shape[1]
    assert D1 == 384 and D2 == 384, "kernel is specialised for 3 x 128 latent"
    BM, BI, ISPLIT, nw, ns, per_sm, sk = config if config is not None else _pick(M)
    assert N1 % (ISPLIT * BI) == 0
    dev = latent.device
    MK = M * top_k
    cnt, sync = _counters(dev, E)
    tw = torch.empty(MK, dtype=torch.float32, device=dev)
    rows = torch.empty(E, M, dtype=torch.int32, device=dev)
    part = _zeroed_partials(dev, ISPLIT * MK * D2).view(ISPLIT, MK, D2)
    args = (
        router_logits,
        e_score_bias,
        latent,
        w13,
        w2,
        part,
        tw,
        rows,
        cnt,
        sync,
        M,
        router_logits.stride(0),
        latent.stride(0),
        M,
    )
    meta = dict(
        E=E,
        TOPK=top_k,
        KP=triton.next_power_of_2(top_k),
        RB=4,
        D1=D1,
        N1=N1,
        D2=D2,
        BM=BM,
        BI=BI,
        ISPLIT=ISPLIT,
        SK=sk,
        num_warps=nw,
        num_stages=ns,
    )
    ckey = ("k", latent.dtype, router_logits.dtype, dev, BM, BI, ISPLIT, nw, ns, sk)
    if ckey not in _scratch:
        _scratch[ckey] = _fused_moe_kernel.warmup(*args, 1, grid=(1,), **meta)
    sms, resident = _resident_per_sm(_scratch[ckey], dev)
    if resident < 1:
        raise RuntimeError("fused MoE config does not fit on one SM")
    max_tiles = (E + triton.cdiv(MK, BM)) * ISPLIT  # >= steps / (IS // BI)
    nprog = min(sms * min(per_sm, resident), max_tiles)
    _fused_moe_kernel[(nprog,)](*args, nprog, **meta)
    return part, ISPLIT


def olala_moe_fused(
    hidden: torch.Tensor,  # [M, hidden] bf16
    router_logits: torch.Tensor,  # [M, E] fp32
    e_score_bias: torch.Tensor,  # [E] fp32
    w_in: torch.Tensor,  # [latent + shared_inter, hidden]  (fc1 ‖ shared_up)
    w_out: torch.Tensor,  # [hidden, latent + shared_inter]  (fc2 ‖ shared_down)
    w13: torch.Tensor,  # [E, inter, latent]
    w2: torch.Tensor,  # [E, latent, inter]
    top_k: int,
    routed_scale: float,
    config=None,
    x_in: torch.Tensor | None = None,
) -> torch.Tensor:
    """shared(x) + fc2(scale * sum_k w_k * E_k(fc1(x))) with the fused routed kernel.

    ``x_in`` ([M, D1 + S], bf16 or fp32, any row stride) is fc1(x) ‖ shared_up(x)
    when the caller already computed it; it is rounded to bf16 on load.
    """
    M = hidden.shape[0]
    D1 = w13.shape[2]
    D2 = w2.shape[1]
    S = w_in.shape[0] - D1
    if x_in is None:
        x_in = torch.nn.functional.linear(hidden, w_in)
    part, isplit = moe_routed_fused(
        x_in[:, :D1], router_logits, e_score_bias, w13, w2, top_k, config
    )
    cat = torch.empty(M, D2 + S, dtype=hidden.dtype, device=hidden.device)
    _fused_epilogue_kernel[(M, (D2 + S) // 128)](
        part,
        x_in[:, D1:],
        cat,
        routed_scale,
        x_in.stride(0),
        M * top_k,
        D2=D2,
        S=S,
        BS=128,
        K=top_k,
        ISPLIT=isplit,
        num_warps=1,
    )
    return torch.nn.functional.linear(cat, w_out)
