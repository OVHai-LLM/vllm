// Olala Differential-TPA decode attention over the TPA-factorized paged KV cache, Blackwell tcgen05 edition (sm_100 / sm_103).
// Same cache layout, launch contract and partial-output format as tpa_factor_decode.cu (sm_90a) and _sm80.cu; built JIT by
// tpa_factor.py.
//
// Blackwell runs the legacy mma.sync path at a quarter of the tcgen05 rate (~0.56 PFLOPS on B300), which is where the factor
// decode is bound once its scalar work is gone, so both products go to the 5th-gen tensor cores with their accumulators in
// tensor memory, warp-specialized as in FlashAttention-4:
//   * TMA warp (warp 7): the sm_90a kernel's producer. 13-token tiles + halo, 128B-swizzled K-major Bk / MN-major Bv
//     tiles and the C|D coefficient boxes into an NBUF-stage mbarrier ring; the split's block ids are cached in smem.
//   * MMA warp (warp 3, one thread): qB = Q Bk^T (UMMA M=64 N=56 K=128, Q padded to 64 heads in smem) into one of two
//     TMEM qB buffers and O += W Bv (UMMA M=64 N=2x64 K=64), each issued as soon as its inputs are ready, so the softmax
//     of tile i overlaps the qB product of tile i+1 and the W Bv product of tile i-1. tcgen05.commit releases buffers.
//   * softmax warps (0-2 and 4-6): one thread per query head (M=64 accumulators live in TMEM lanes 0-15 of each
//     sub-partition; warps w and w+4 share sub-partition w and split a tile's tokens 7 + 6), tcgen05.ld of the qB row,
//     rank-4 score assembly, softcap, online softmax with the lazy rescale of FA4 (the running max moves only when a
//     tile raises it by > 8 nats; the O row is then rescaled in TMEM), and the W row written to smem in the
//     128B-swizzled K-major layout of the second product.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <unordered_map>

#ifndef TPA_NBUF
#define TPA_NBUF 2
#endif

namespace {
constexpr int HQ = 48, HQP = 64, HKV = 12, R = 4, D = 128, GQ = HQ / HKV;
constexpr int T = 13, NTOK = T + 1, NCOL = NTOK * R;    // 56 columns c = tokidx*4 + r (tokidx 0 = t0-1)
constexpr int ROWP = 608, OFF_C = 512, NCOEF = 96;       // per token and plane: C (12x4) | D (12x4)
constexpr int NBUF = TPA_NBUF;
// warps 0-2 and 4-6: softmax (warps w and w + 4 share TMEM sub-partition w, i.e. the same 16 heads, and split each
// tile's tokens 7 + 6), warp 3: MMA, warp 7: TMA
constexpr int NSOFT = 6, THREADS = 8 * 32;
constexpr int MMA_WARP = 3, TMA_WARP = 7;
constexpr int T_LO = 7;   // tokens 0..6 -> warps 0-2, 7..12 -> warps 4-6
constexpr int HALF_BYTES = 16 * R * 128;                 // 64 rows x 128 B (a 64-element k-half / n-half)
constexpr unsigned TILE_TX = 2u * (NTOK * 2 * 512 + T * NCOEF * 2);
constexpr int TMEM_COLS = 256;                           // qB[2] (2 x 64 cols) + O (128 cols)
constexpr int TM_QB = 0, TM_O = 128;
#ifndef TPA_RESCALE_THRESHOLD
#define TPA_RESCALE_THRESHOLD 8.f
#endif
constexpr float RESCALE_THRESHOLD = TPA_RESCALE_THRESHOLD;   // nats
constexpr int NBLK_CACHE = 256;

struct __align__(1024) Smem {
  char q[2][HALF_BYTES];          // Q (64 x 128), K-major SW128, per 64-element k-half; heads 48..63 zero
  char w[2][HALF_BYTES];          // W tiles (64 x 64), K-major SW128, double buffer; rows 48..63 zero
  char bk[NBUF][2][HALF_BYTES];   // [buf][k half] K-major SW128: row n = tok*4 + r; rows 56..63 zero
  char bv[NBUF][2][HALF_BYTES];   // [buf][n half] MN-major SW128: row k = tok*4 + r; rows 56..63 zero
  __nv_bfloat16 ck[NBUF][T * NCOEF + 96];   // tokens t = 0..12: [Ck 48 | Dk 48]
  __nv_bfloat16 cv[NBUF][T * NCOEF + 96];
  unsigned long long full[NBUF], empty[NBUF], s_full[2], s_free[2], w_full[2], o_done[2];
  uint32_t tmem_base;
  int blk_first, nblk;
  int blocks[NBLK_CACHE];   // block ids of this split (the producer would otherwise wait on a global load per tile)
  float xmax[2][HQP], xp7[HQP], xl[HQP];   // softmax warp-pair exchange
};

__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(void* mbar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(void* mbar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(void* mbar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(void* mbar, unsigned parity) {
  unsigned done = 0;
  while (!done) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(smem_u32(mbar)), "r"(parity) : "memory");
  }
}
__device__ __forceinline__ bool mbar_test(void* mbar, unsigned parity) {
  unsigned done;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(done) : "r"(smem_u32(mbar)), "r"(parity) : "memory");
  return done != 0;
}
__device__ __forceinline__ void tma4d(void* dst, const CUtensorMap* tmap, int c0, int c1, int c2, int c3, void* mbar) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tma3d(void* dst, const CUtensorMap* tmap, int c0, int c1, int c2, void* mbar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tma_bulk_g2s(void* dst, const void* src, unsigned bytes, void* mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// ---- tcgen05 helpers
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
// smem matrix descriptor (sm_100): addr, LBO, SBO (>>4), version 1, 128B swizzle
__device__ __forceinline__ uint64_t umma_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  uint64_t d = (uint64_t)((saddr & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
// instruction descriptor: f32 accumulate, bf16 A/B, A K-major, B K- or MN-major
__host__ __device__ constexpr uint32_t umma_idesc(int M, int N, int b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
__device__ __forceinline__ void umma(uint32_t tmem_d, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               :: "r"(tmem_d), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
__device__ __forceinline__ void umma_commit(void* mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" :: "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float* v) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float* v) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld8(uint32_t taddr, float* v) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
__device__ __forceinline__ void tmem_st32(uint32_t taddr, const float* v) {
  const uint32_t* r = reinterpret_cast<const uint32_t*>(v);
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
               "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};\n"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]),
                  "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]),
                  "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }

struct TmaMaps { CUtensorMap k13, k1, v13, v1, ck13, cv13; };

struct BlockLookup {
  const int* __restrict__ btb; const int* cache; int first, n;
  __device__ __forceinline__ void operator()(int t, int BS, int& slot, int& blk) const {
    const int i = t / BS - first;
    blk = (i >= 0 && i < n) ? cache[i] : __ldg(btb + t / BS);
    slot = t % BS;
  }
};

// sm_90a kernel's producer: one wide TMA set per tile (13 tokens of one block + the halo), or one lane per token at the
// first / last tile with the tokens outside [0, seqlen) zero-filled.
__device__ __forceinline__ void producer_issue(Smem& S, int buf, const TmaMaps& M, const __nv_bfloat16* __restrict__ kpl, const __nv_bfloat16* __restrict__ vpl,
                                               long long bstride, const BlockLookup& locate, int BS, int seqlen, int t0, int lane) {
  const bool wide = (t0 >= 1) && (t0 + T <= seqlen);
  if (wide) {
    if (lane == 0) mbar_arrive_expect_tx(&S.full[buf], TILE_TX);
    __syncwarp();
    if (lane < 10) {
      int s0, b0, sp, bp; locate(t0, BS, s0, b0); locate(t0 - 1, BS, sp, bp);
      switch (lane) {
        case 0: tma4d(S.bk[buf][0] + 512, &M.k13, 0, 0, s0, b0, &S.full[buf]); break;
        case 1: tma4d(S.bk[buf][1] + 512, &M.k13, 64, 0, s0, b0, &S.full[buf]); break;
        case 2: tma4d(S.bv[buf][0] + 512, &M.v13, 0, 0, s0, b0, &S.full[buf]); break;
        case 3: tma4d(S.bv[buf][1] + 512, &M.v13, 64, 0, s0, b0, &S.full[buf]); break;
        case 4: tma3d(S.ck[buf], &M.ck13, 0, s0, b0, &S.full[buf]); break;
        case 5: tma3d(S.cv[buf], &M.cv13, 0, s0, b0, &S.full[buf]); break;
        case 6: tma4d(S.bk[buf][0], &M.k1, 0, 0, sp, bp, &S.full[buf]); break;
        case 7: tma4d(S.bk[buf][1], &M.k1, 64, 0, sp, bp, &S.full[buf]); break;
        case 8: tma4d(S.bv[buf][0], &M.v1, 0, 0, sp, bp, &S.full[buf]); break;
        case 9: tma4d(S.bv[buf][1], &M.v1, 64, 0, sp, bp, &S.full[buf]); break;
      }
    }
    return;
  }
  const int tok = lane, t = t0 - 1 + tok;
  const bool mine = lane < NTOK, valid = mine && (t >= 0) && (t < seqlen);
  const unsigned nvalid = __popc(__ballot_sync(0xffffffffu, valid));
  const unsigned nvalid_c = __popc(__ballot_sync(0xffffffffu, valid && tok >= 1));
  if (mine && !valid) {
    for (int h = 0; h < 2; ++h) for (int c = 0; c < 32; ++c) {
      *reinterpret_cast<uint4*>(S.bk[buf][h] + tok * 512 + c * 16) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(S.bv[buf][h] + tok * 512 + c * 16) = make_uint4(0, 0, 0, 0);
    }
    if (tok >= 1) {
      __nv_bfloat16* zk = S.ck[buf] + (tok - 1) * NCOEF; __nv_bfloat16* zv = S.cv[buf] + (tok - 1) * NCOEF;
      for (int c = 0; c < NCOEF / 8; ++c) { *reinterpret_cast<uint4*>(zk + c * 8) = make_uint4(0, 0, 0, 0); *reinterpret_cast<uint4*>(zv + c * 8) = make_uint4(0, 0, 0, 0); }
    }
    fence_proxy_async();   // the tensor cores read B through the async proxy
  }
  __syncwarp();
  if (lane == 0) mbar_arrive_expect_tx(&S.full[buf], nvalid * 2u * 2 * 512 + nvalid_c * 2u * NCOEF * 2);
  __syncwarp();
  if (mine && valid) {
    int s, bl; locate(t, BS, s, bl);
    tma4d(S.bk[buf][0] + tok * 512, &M.k1, 0, 0, s, bl, &S.full[buf]);
    tma4d(S.bk[buf][1] + tok * 512, &M.k1, 64, 0, s, bl, &S.full[buf]);
    tma4d(S.bv[buf][0] + tok * 512, &M.v1, 0, 0, s, bl, &S.full[buf]);
    tma4d(S.bv[buf][1] + tok * 512, &M.v1, 64, 0, s, bl, &S.full[buf]);
    if (tok >= 1) {
      const size_t grow = (size_t)bl * bstride + (size_t)s * ROWP;
      tma_bulk_g2s(S.ck[buf] + (tok - 1) * NCOEF, kpl + grow + OFF_C, NCOEF * 2, &S.full[buf]);
      tma_bulk_g2s(S.cv[buf] + (tok - 1) * NCOEF, vpl + grow + OFF_C, NCOEF * 2, &S.full[buf]);
    }
  }
}

__device__ __forceinline__ float4 ld_coef4(const __nv_bfloat16* p) {   // 4 bf16 (8 B aligned) -> float4
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ void pair_bar(int quad) { asm volatile("bar.sync %0, 64;\n" :: "r"(1 + quad) : "memory"); }

// One softmax warp: thread (quad, lane < 16) owns head quad*16 + lane (lanes 16..31 shadow lane - 16 and store nothing).
// HALF 0 scores tokens 0..6 of each tile and writes W columns of tokidx 0..7; HALF 1 tokens 7..12 and tokidx 8..15.
// The pair exchanges its tile maxima (both take the same rescale decision), p[7] (needed by HALF 0's tokidx 7) and,
// at the end, its partial row sums. HALF 0 owns the O row (rescale in TMEM, final partials).
template <int HALF>
__device__ __forceinline__ void softmax_warp(Smem& S, uint32_t tmem, int quad, int lane, int ntiles, int start, int stop,
                                             float sm_scale, float softcap, float* __restrict__ part_o, float* __restrict__ part_m,
                                             float* __restrict__ part_l, size_t base) {
  constexpr int TB = HALF ? T_LO : 0, NT = HALF ? T - T_LO : T_LO, C0 = HALF ? 24 : 0;   // qv[i] = qB column C0 + i
  const bool live = lane < 16;
  const int h = quad * 16 + (lane & 15), g = h / GQ;
  const uint32_t trow = tmem + ((uint32_t)(quad * 32) << 16);
  const float inv_cap = softcap > 0.f ? 1.f / softcap : 0.f;
  float m = -CUDART_INF_F, l = 0.f;
  for (int it = 0; it < ntiles; ++it) {
    const int buf = it % NBUF, t0 = start + it * T;
    mbar_wait(&S.full[buf], (it / NBUF) & 1);
    mbar_wait(&S.s_full[it % 2], (it / 2) & 1);
    tc_fence_after();
    float qv[32];
    tmem_ld32(trow + TM_QB + (it % 2) * 64 + C0, qv);
    tmem_wait_ld();
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(&S.s_free[it % 2]);

    // scores s_t = sum_r Ck[t,g,r] qB[(t+1)*4+r] + Dk[t,g,r] qB[t*4+r]
    const __nv_bfloat16* ck = S.ck[buf];
    float pl[8];
    float mt = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int t = TB + j, a = (t + 1) * 4 - C0, bq = t * 4 - C0;
      const float4 c = ld_coef4(ck + t * NCOEF + g * R), d = ld_coef4(ck + t * NCOEF + HKV * R + g * R);
      float sc = c.x * qv[a] + c.y * qv[a + 1] + c.z * qv[a + 2] + c.w * qv[a + 3]
               + d.x * qv[bq] + d.y * qv[bq + 1] + d.z * qv[bq + 2] + d.w * qv[bq + 3];
      sc *= sm_scale;
      if (softcap > 0.f) sc = softcap * tanhf(sc * inv_cap);
      sc = (t0 + t < stop) ? sc : -CUDART_INF_F;
      pl[j] = sc;
      mt = fmaxf(mt, sc);
    }
    if (live) S.xmax[HALF][h] = mt;
    pair_bar(quad);
    mt = fmaxf(mt, S.xmax[1 - HALF][h]);
    // lazy rescale: keep the running max unless this tile raises it by more than the threshold
    const bool up = mt > m + RESCALE_THRESHOLD;   // also when m == -inf
    const float m_new = up ? mt : m;
    const float alpha = up ? ((m == -CUDART_INF_F) ? 0.f : __expf(m - m_new)) : 1.f;
    float ls = 0.f;
#pragma unroll
    for (int j = 0; j < NT; ++j) { pl[j] = (pl[j] == -CUDART_INF_F) ? 0.f : __expf(pl[j] - m_new); ls += pl[j]; }
#pragma unroll
    for (int j = NT; j < 8; ++j) pl[j] = 0.f;
    l = l * alpha + ls; m = m_new;
    if (HALF == 1 && live) S.xp7[h] = pl[0];
    pair_bar(quad);
    if (HALF == 0) pl[7] = S.xp7[h];

    if (it >= 2) mbar_wait(&S.o_done[it % 2], ((it - 2) / 2) & 1);   // W buffer it % 2 free again
    if (HALF == 0 && it >= 1 && __any_sync(0xffffffffu, live && alpha != 1.f)) {   // O row *= alpha, after W Bv of tile it-1
      mbar_wait(&S.o_done[(it - 1) % 2], ((it - 1) / 2) & 1);
      tc_fence_after();
      const float a = live ? alpha : 1.f;
#pragma unroll 1
      for (int c4 = 0; c4 < D / 32; ++c4) {
        float o[32];
        tmem_ld32(trow + TM_O + c4 * 32, o);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 32; ++k) o[k] *= a;
        tmem_st32(trow + TM_O + c4 * 32, o);
      }
      tmem_wait_st();
    }
    // W row: W[(tokidx, r)] = p[tokidx-1] Cv[tokidx-1, g, r] + p[tokidx] Dv[tokidx, g, r]; columns 56..63 zero
    const __nv_bfloat16* cv = S.cv[buf];
    char* wrow = S.w[it % 2] + h * 128;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = jj + 4 * HALF;   // 16 B chunk j = tokidx 2j, 2j+1
      float w[8];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int tokidx = 2 * j + u;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tokidx >= 1 && tokidx <= T) {
          const float4 c = ld_coef4(cv + (tokidx - 1) * NCOEF + g * R);
          const float pp = pl[tokidx - 1 - TB];
          acc.x += pp * c.x; acc.y += pp * c.y; acc.z += pp * c.z; acc.w += pp * c.w;
        }
        if (tokidx <= T - 1) {
          const float4 d = ld_coef4(cv + tokidx * NCOEF + HKV * R + g * R);
          const float pp = pl[tokidx - TB];
          acc.x += pp * d.x; acc.y += pp * d.y; acc.z += pp * d.z; acc.w += pp * d.w;
        }
        w[4 * u + 0] = acc.x; w[4 * u + 1] = acc.y; w[4 * u + 2] = acc.z; w[4 * u + 3] = acc.w;
      }
      if (live)
        *reinterpret_cast<uint4*>(wrow + ((j ^ (h & 7)) * 16)) =
            make_uint4(pack_bf16(w[0], w[1]), pack_bf16(w[2], w[3]), pack_bf16(w[4], w[5]), pack_bf16(w[6], w[7]));
    }
    fence_proxy_async();
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(&S.w_full[it % 2]);
  }
  // ---- partials (HALF 0): unnormalized O row, running max and the pair's row sum
  if (HALF == 1 && live) S.xl[h] = l;
  pair_bar(quad);
  if (HALF == 1) return;
  l += S.xl[h];
  mbar_wait(&S.o_done[(ntiles - 1) % 2], ((ntiles - 1) / 2) & 1);
  tc_fence_after();
#pragma unroll 1
  for (int c4 = 0; c4 < D / 32; ++c4) {
    float o[32];
    tmem_ld32(trow + TM_O + c4 * 32, o);
    tmem_wait_ld();
    if (live) {
      float4* dst = reinterpret_cast<float4*>(part_o + (base + h) * D + c4 * 32);
#pragma unroll
      for (int k = 0; k < 8; ++k) dst[k] = make_float4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
    }
  }
  if (live) { part_m[base + h] = m; part_l[base + h] = l; }
}

__global__ void __launch_bounds__(THREADS, NBUF <= 2 ? 2 : 1)
tpa_factor_decode_tc_kernel(const __grid_constant__ TmaMaps maps, const __nv_bfloat16* __restrict__ kpl, const __nv_bfloat16* __restrict__ vpl, long long bstride,
                            const __nv_bfloat16* __restrict__ q, const int* __restrict__ bt, int bt_stride, const int* __restrict__ seqlens,
                            float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
                            float sm_scale, float softcap, const int* __restrict__ tok_per_split_ptr, int BS) {
  extern __shared__ __align__(1024) char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int tok_per_split = *tok_per_split_ptr;
  const int b = blockIdx.x, sp = blockIdx.y, nsplit = gridDim.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int seqlen = seqlens[b];
  const int start = sp * tok_per_split, stop = min(start + tok_per_split, seqlen);
  const size_t base = ((size_t)b * nsplit + sp) * HQP;
  if (stop <= start) {
    if (tid < HQP) { part_m[base + tid] = -CUDART_INF_F; part_l[base + tid] = 0.f; }
    return;
  }
  const int* btb = bt + (size_t)b * bt_stride;

  // ---- setup: TMEM, barriers, Q (swizzled K-major), structural zeros
  if (warp == MMA_WARP) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(smem_u32(&S.tmem_base)), "n"(TMEM_COLS));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
  }
  if (tid == 0) {
    for (int i = 0; i < NBUF; ++i) { mbar_init(&S.full[i], 1); mbar_init(&S.empty[i], 1); }
    for (int i = 0; i < 2; ++i) { mbar_init(&S.s_full[i], 1); mbar_init(&S.s_free[i], NSOFT); mbar_init(&S.w_full[i], NSOFT); mbar_init(&S.o_done[i], 1); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  {
    const __nv_bfloat16* qb = q + (size_t)b * HQ * D;
    for (int i = tid; i < HQP * 16; i += THREADS) {
      const int r = i / 16, h = (i / 8) % 2, j = i % 8;
      uint4 v = r < HQ ? *reinterpret_cast<const uint4*>(qb + r * D + h * 64 + j * 8) : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(S.q[h] + r * 128 + ((j ^ (r & 7)) * 16)) = v;
    }
    for (int i = tid; i < 2 * (HQP - HQ) * 8; i += THREADS)   // W rows 48..63
      *reinterpret_cast<uint4*>(S.w[i / ((HQP - HQ) * 8)] + HQ * 128 + (i % ((HQP - HQ) * 8)) * 16) = make_uint4(0, 0, 0, 0);
    for (int i = tid; i < NBUF * 2 * 2 * 64; i += THREADS) {   // B rows 56..63 (never written by TMA)
      const int buf = i / 256, rem = i % 256, pl = rem / 128, h = (rem / 64) % 2, c = rem % 64;
      *reinterpret_cast<uint4*>((pl == 0 ? S.bk[buf][h] : S.bv[buf][h]) + 7168 + c * 16) = make_uint4(0, 0, 0, 0);
    }
  }
  {
    const int bfirst = max(start - 1, 0) / BS, nb = min((max(stop, 1) - 1) / BS - bfirst + 1, NBLK_CACHE);
    if (tid == 0) { S.blk_first = bfirst; S.nblk = nb; }
    for (int i = tid; i < nb; i += THREADS) S.blocks[i] = btb[bfirst + i];
  }
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = S.tmem_base;
  const int ntiles = (stop - start + T - 1) / T;

  if (warp == TMA_WARP) {
    const BlockLookup locate{btb, S.blocks, S.blk_first, S.nblk};
    for (int it = 0; it < ntiles; ++it) {
      const int buf = it % NBUF;
      if (it >= NBUF) mbar_wait(&S.empty[buf], ((it / NBUF) + 1) & 1);
      producer_issue(S, buf, maps, kpl, vpl, bstride, locate, BS, seqlen, start + it * T, lane);
    }
  } else if (warp == MMA_WARP) {
    if (lane == 0) {
      constexpr uint32_t ID1 = umma_idesc(64, NCOL, 0), ID2 = umma_idesc(64, 64, 1);
      // issue each product as soon as its inputs are ready (qB(i) needs tile i and a free qB buffer, W Bv(j) needs W(j)),
      // so a stage is handed back to the TMA warp right after its W Bv product instead of after the next tile lands
      int n1 = 0, n2 = 0;
      while (n2 < ntiles) {
        if (n1 < ntiles && n1 - n2 < 2 && mbar_test(&S.full[n1 % NBUF], (n1 / NBUF) & 1) &&
            (n1 < 2 || mbar_test(&S.s_free[n1 % 2], ((n1 - 2) / 2) & 1))) {
          const int buf = n1 % NBUF;
          tc_fence_after();
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int k = 0; k < 4; ++k)
              umma(tmem + TM_QB + (n1 % 2) * 64, umma_desc(smem_u32(S.q[h]) + k * 32, 16, 1024),
                   umma_desc(smem_u32(S.bk[buf][h]) + k * 32, 16, 1024), ID1, (h | k) ? 1u : 0u);
          umma_commit(&S.s_full[n1 % 2]);
          ++n1;
        }
        if (n2 < n1 && mbar_test(&S.w_full[n2 % 2], (n2 / 2) & 1)) {
          const int bj = n2 % NBUF;
          tc_fence_after();
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const uint64_t ad = umma_desc(smem_u32(S.w[n2 % 2]) + k * 32, 16, 1024);
#pragma unroll
            for (int h = 0; h < 2; ++h)
              umma(tmem + TM_O + h * 64, ad, umma_desc(smem_u32(S.bv[bj][h]) + k * 2048, 8192, 1024), ID2, (n2 > 0 || k > 0) ? 1u : 0u);
          }
          umma_commit(&S.o_done[n2 % 2]);
          umma_commit(&S.empty[bj]);
          ++n2;
        }
      }
    }
    __syncwarp();
  } else if (warp < 4) {
    softmax_warp<0>(S, tmem, warp, lane, ntiles, start, stop, sm_scale, softcap, part_o, part_m, part_l, base);
  } else {
    softmax_warp<1>(S, tmem, warp - 4, lane, ntiles, start, stop, sm_scale, softcap, part_o, part_m, part_l, base);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == MMA_WARP) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tmem), "n"(TMEM_COLS));
  }
}
}  // namespace

// ---------------------------------------------------------------- host (tensor maps as in tpa_factor_decode.cu)
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn get_encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qres; void* p = nullptr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qres);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<EncodeTiledFn>(p);
  }
  return fn;
}
static CUtensorMap make_factor_map(const void* base, int64_t bs, int64_t nb, int64_t bstride, int box_tok) {
  CUtensorMap m;
  cuuint64_t gdim[4] = {(cuuint64_t)D, (cuuint64_t)R, (cuuint64_t)bs, (cuuint64_t)nb};
  cuuint64_t gstride[3] = {(cuuint64_t)D * 2, (cuuint64_t)ROWP * 2, (cuuint64_t)bstride * 2};
  cuuint32_t box[4] = {64, (cuuint32_t)R, (cuuint32_t)box_tok, 1};
  cuuint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = get_encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(base), gdim, gstride, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(factor) failed: ", (int)r);
  return m;
}
static CUtensorMap make_coef_map(const void* base, int64_t bs, int64_t nb, int64_t bstride, int box_tok) {
  CUtensorMap m;
  cuuint64_t gdim[3] = {(cuuint64_t)NCOEF, (cuuint64_t)bs, (cuuint64_t)nb};
  cuuint64_t gstride[2] = {(cuuint64_t)ROWP * 2, (cuuint64_t)bstride * 2};
  cuuint32_t box[3] = {(cuuint32_t)NCOEF, (cuuint32_t)box_tok, 1};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), gdim, gstride, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(coef) failed: ", (int)r);
  return m;
}

void tpa_factor_decode_tc_launch(torch::Tensor q, torch::Tensor kplane, torch::Tensor vplane, torch::Tensor block_table, torch::Tensor seq_lens,
                                 torch::Tensor tok_per_split, torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
                                 double sm_scale, double softcap, int64_t block_size) {
  const int B = q.size(0), SPLIT = part_o.size(1);
  TORCH_CHECK(kplane.dim() == 3 && kplane.size(2) == ROWP && kplane.stride(2) == 1 && kplane.stride(1) == ROWP && q.is_contiguous());
  TORCH_CHECK(vplane.stride(0) == kplane.stride(0) && vplane.stride(1) == ROWP && kplane.size(1) == block_size && block_size % T == 0);
  TORCH_CHECK(block_table.dtype() == torch::kInt32 && seq_lens.dtype() == torch::kInt32 && tok_per_split.dtype() == torch::kInt32);
  TORCH_CHECK(part_o.size(2) == HQP && part_o.size(3) == D && part_o.is_contiguous() && part_m.is_contiguous() && part_l.is_contiguous());
  // keyed by everything the maps encode: a freed plane's address can come back with another shape
  struct Key { const void* k; const void* v; int64_t nb, bs, bstr;
               bool operator==(const Key& o) const { return k == o.k && v == o.v && nb == o.nb && bs == o.bs && bstr == o.bstr; } };
  struct KeyHash { size_t operator()(const Key& x) const { return std::hash<const void*>()(x.k) ^ (size_t)(x.nb * 1000003 + x.bs * 131 + x.bstr); } };
  static std::unordered_map<Key, TmaMaps, KeyHash> maps;
  const void* kb = kplane.data_ptr(); const void* vb = vplane.data_ptr();
  const Key key{kb, vb, kplane.size(0), block_size, kplane.stride(0)};
  auto it = maps.find(key);
  if (it == maps.end()) {
    TmaMaps M;
    const int64_t nb = kplane.size(0), bs = block_size, bstr = kplane.stride(0);
    M.k13 = make_factor_map(kb, bs, nb, bstr, T); M.k1 = make_factor_map(kb, bs, nb, bstr, 1);
    M.v13 = make_factor_map(vb, bs, nb, bstr, T); M.v1 = make_factor_map(vb, bs, nb, bstr, 1);
    M.ck13 = make_coef_map(static_cast<const __nv_bfloat16*>(kb) + OFF_C, bs, nb, bstr, T);
    M.cv13 = make_coef_map(static_cast<const __nv_bfloat16*>(vb) + OFF_C, bs, nb, bstr, T);
    it = maps.emplace(key, M).first;
  }
  static bool attr_set = false;
  const int smem = (int)sizeof(Smem) + 1024;
  if (!attr_set) {
    cudaFuncSetAttribute(tpa_factor_decode_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(tpa_factor_decode_tc_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    attr_set = true;
  }
  tpa_factor_decode_tc_kernel<<<dim3(B, SPLIT), THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      it->second, reinterpret_cast<const __nv_bfloat16*>(kb), reinterpret_cast<const __nv_bfloat16*>(vb), (long long)kplane.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), block_table.data_ptr<int>(), (int)block_table.stride(0), seq_lens.data_ptr<int>(),
      part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(), (float)sm_scale, (float)softcap, tok_per_split.data_ptr<int>(), (int)block_size);
}
int64_t tpa_factor_tc_smem_bytes() { return sizeof(Smem) + 1024; }
int64_t tpa_factor_tc_occupancy() {
  const int smem = (int)sizeof(Smem) + 1024;
  cudaFuncSetAttribute(tpa_factor_decode_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(tpa_factor_decode_tc_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  int nb = -1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, tpa_factor_decode_tc_kernel, THREADS, smem);
  return nb;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &tpa_factor_decode_tc_launch); m.def("smem_bytes", &tpa_factor_tc_smem_bytes);
  m.def("occupancy", &tpa_factor_tc_occupancy); m.attr("TILE") = T; m.attr("INNER_TILE") = T;
}
