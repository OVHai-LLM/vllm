// Olala Mamba-3 MIMO decode step for large batches, preamble fused. Replaces _decode_preamble_bc_kernel +
// mamba3_step_f32_pipe_kernel after the in_proj GEMM; reads the raw in_proj output zb = [in_proj | in_proj_dyn](u).
//
// Bound by the state (64 x 128 per head and token, read and rewritten). With HBM saturated every dependent global
// access waits ~1-2 us in the memory queues, so nothing global sits on an item's critical path: each CTA owns one head
// (constants loaded once into shared memory), its slots are read once, and everything item k + 1 needs (state tile,
// in_proj head row, B / C / angle rows, previous B', angles, previous x) is fetched with cp.async while item k runs.
// The state tile is consumed straight into the mma accumulators (seeded by alpha h), one column half at a time (32
// accumulator registers -> 4 CTAs / SM); h_new goes from the accumulators to the pool. The S columns are permuted
// inside each 16-column group so that a lane's four columns are contiguous (one vector access per row and group);
// B' / C' are laid out in the same order, so the update is unchanged and the C dot reduces over the same groups.
// Warp r preps row r of B / C and owns row r of k_pool. Three barriers per item (HC_DB=1: two tiles for 16-bit
// formats and no mid-item barrier; neutral, off).
//
// Per column half the C dot is issued before the requantization so its HMMAs cover the row-max latency.
//
// State formats: fp32 (bit-identical to mamba3_step_split); Q16 = int16 mantissas stored offset binary (u = q + 32768)
// with one fp32 scale per half row (64 values), h = q * scale, |q| <= 32767: 16.5 bits per value, relative storage
// error ~3e-5 (bf16: 1.6e-3); and bf16 (reference for what a plain 16-bit state costs). Q16 decode: one PRMT builds
// the float 2^23 + u for two mantissas, one FADD2 removes the magic; the accumulators of a row half are kept in units of
// alpha * scale ("q units"), so the tile is consumed as is and the requantization needs no scale multiply; encode:
// scale ~ max / 32767 (approximate reciprocal), u = low 16 bits of fma(h, 1 / scale, 1.5 * 2^23 + 32768) (FFMA2).
// B300, graphed, preamble + step per layer at B = 64 / 128 / 256: fp32 41 / 73 / 140 us (default preamble + pipe
// step: 43 / 78 / 149), Q16 24 / 46 / 82 us. Q16 decode NLL equal to the fp32 state over 6 x 3000 tokens.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <unordered_map>

namespace {
constexpr int H = 48, D = 64, S = 128, R = 4, NA = 32;
constexpr int NZX = 2 * D + 3;          // in_proj rows per head: z (D) | x (D) | dt | A | trap
constexpr int N1 = H * NZX;             // 6288
constexpr int NDYN = 2 * R * S + NA;    // 1056: B (R x S) | C (R x S) | angles
constexpr int THREADS = 128;
constexpr float QU_MIN = 5.421010862427522e-20f;   // 2^-64: smallest Q16 accumulator unit (see the item loop)
constexpr int LDB = S + 8;
constexpr int HROW = 144;               // 16B-aligned superset of a head's in_proj row (18 chunks)
constexpr int MAXK = 256;               // items per CTA

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(s)), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma1688(float* c, const uint32_t* a, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(b));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ void split3_x2(float x0, float x1, uint32_t& hi, uint32_t& mid, uint32_t& lo) {
  hi = pack_bf16(x0, x1);
  const float r0 = x0 - __uint_as_float(hi << 16), r1 = x1 - __uint_as_float(hi & 0xffff0000u);
  mid = pack_bf16(r0, r1);
  lo = pack_bf16(r0 - __uint_as_float(mid << 16), r1 - __uint_as_float(mid & 0xffff0000u));
}
__device__ __forceinline__ void split2_x2(float x0, float x1, uint32_t& hi, uint32_t& lo) {
  hi = pack_bf16(x0, x1);
  lo = pack_bf16(x0 - __uint_as_float(hi << 16), x1 - __uint_as_float(hi & 0xffff0000u));
}
// x = hi + mid (+ lo): 3 bf16 terms (~fp32) for the fp32 state, 2 (16 mantissa bits, below the Q16 storage error) for Q16
template <int NS> __device__ __forceinline__ void splitn_x2(float x0, float x1, uint32_t* t0, uint32_t* t1, uint32_t* t2) {
  if constexpr (NS == 3) split3_x2(x0, x1, *t0, *t1, *t2); else if constexpr (NS == 2) split2_x2(x0, x1, *t0, *t1); else *t0 = pack_bf16(x0, x1);
}
// TMA / bulk async copies (16-bit states): the tile moves as two 64-column boxes, 128B-swizzled in shared memory
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(b)), "r"(n)); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int par) {
  asm volatile("{\n .reg .pred p;\n MBW%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra MBW%=;\n}\n" :: "r"(smem_u32(b)), "r"(par) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_g2s(void* dst, const CUtensorMap* tm, int x, int y, int z, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
               :: "r"(smem_u32(dst)), "l"(tm), "r"(x), "r"(y), "r"(z), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_s2g(const CUtensorMap* tm, int x, int y, int z, const void* src) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4];\n"
               :: "l"(tm), "r"(x), "r"(y), "r"(z), "r"(smem_u32(src)) : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, int bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
// Q16 mantissas are stored offset binary (u = q + 32768): one PRMT builds the float 2^23 + u, minus the magic
// offset-binary rint(v) + 32768 for |v| <= 32767 as the low 16 bits of fl(v + 1.5 * 2^23 + 32768), without F2I
// Blackwell packed fp32x2 arithmetic (one instruction for two lanes of a register pair)
__device__ __forceinline__ float2 fadd2(float2 a, float2 b) {
  unsigned long long d;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(*reinterpret_cast<unsigned long long*>(&a)), "l"(*reinterpret_cast<unsigned long long*>(&b)));
  return *reinterpret_cast<float2*>(&d);
}
__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
  unsigned long long d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(*reinterpret_cast<unsigned long long*>(&a)), "l"(*reinterpret_cast<unsigned long long*>(&b)),
      "l"(*reinterpret_cast<unsigned long long*>(&c)));
  return *reinterpret_cast<float2*>(&d);
}
// the two offset-binary mantissas of a word as exact floats: (2^23 + u) - (2^23 + 32768) for both halves in one FADD2
__device__ __forceinline__ float2 i16x2(uint32_t w) {
  return fadd2(make_float2(__uint_as_float(__byte_perm(w, 0x4b00u, 0x5410)), __uint_as_float(__byte_perm(w, 0x4b00u, 0x5432))),
               make_float2(-8421376.f, -8421376.f));
}
__device__ __forceinline__ float softplus20(float v) { return v > 20.f ? v : logf(1.f + expf(v)); }
// 16-bit states: hardware approximations for the per-item scalars, B / C norms and rotation (relative errors <= 1e-5,
// far below the bf16 rounding of the update operand and of B' / C'); the fp32 path keeps the libm forms (bit-identical
// to mamba3_step_split).
#ifndef HC_REGW
#define HC_REGW 0
#endif
#ifndef HC_VEC
#define HC_VEC 0
#endif
#ifndef HC_FASTMATH
#define HC_FASTMATH 1
#endif
template <bool F> __device__ __forceinline__ float softplus_(float v) { if constexpr (F) return v > 20.f ? v : __logf(1.f + __expf(v)); else return softplus20(v); }
template <bool F> __device__ __forceinline__ float sigmoid_(float v) { if constexpr (F) return __fdividef(1.f, 1.f + __expf(-v)); else return 1.f / (1.f + expf(-v)); }
template <bool F> __device__ __forceinline__ float rsqrt_(float v) { if constexpr (F) return rsqrtf(v); else return 1.f / sqrtf(v); }
// tanh = 1 - 2 / (1 + e^2|x|): absolute error ~1e-7 (the angle increment is tanh * dt * pi, so the recurrence drifts
// by < 1e-8 rad per step, below the fp32 rounding of the angle itself)
template <bool F> __device__ __forceinline__ float tanh_(float v) {
  if constexpr (F) return copysignf(1.f - __fdividef(2.f, 1.f + __expf(2.f * fabsf(v))), v); else return tanhf(v);
}
template <bool F> __device__ __forceinline__ void sincos_(float v, float* s, float* c) { if constexpr (F) __sincosf(v, s, c); else sincosf(v, s, c); }
// mma column of natural state column c: n tile 2 (c >> 4) + ((c & 3) >> 1), in-tile column 2 ((c >> 2) & 3) + (c & 1)
__device__ __forceinline__ int perm(int c) { return (c & ~15) + ((c & 2) << 2) + ((c & 12) >> 1) + (c & 1); }
// fp32 tile: 16B chunk ch (0..31) of row p; odd rows swap chunk halves of each 8-chunk group (a float4 phase = rows
// p, p+1 x 4 chunks -> 32 distinct banks)
__device__ __forceinline__ int swz32(int p, int ch) { return p * 32 + (ch ^ ((p & 1) << 2)); }
// int16 tile: 32B unit u (0..7) of row p moved to u ^ (p & 3) (an 8B phase = rows p..p+3 x 32 B -> distinct banks);
// returns the 16B chunk index of half `hf` of unit u
__device__ __forceinline__ int swz16(int p, int u, int hf) { return p * 16 + ((u ^ (p & 3)) << 1) + hf; }

struct Args {
  const __nv_bfloat16* zb; long long zb_stride; int B;
  void* ssm; float* ssc; __nv_bfloat16* kp; __nv_bfloat16* vp; float* angle; int P;
  long long ssm_sst, ssc_sst, kp_sst, vp_sst, ang_sst;
  const int* slots;
  const float* dt_bias; float a_floor; const float* wb; const float* wc; float eps;
  const float* Dp; const float* xproj; const float* zproj; const float* outproj; const float* bias_q; const float* bias_k;
  __nv_bfloat16* y; long long y_stride;
  int pdl;
};

struct __align__(16) Small {
  __nv_bfloat16 dyn[NDYN];              // token's B | C | angle rows
  __nv_bfloat16 hrow[HROW];             // head's z | x | dt | A | trap (from offset hoff)
  __nv_bfloat16 kprev[R * S];
  float angst[NA];
  __nv_bfloat16 xs[D];
};
// state formats: 0 fp32, 1 Q16, 2 bf16 (reference: what a plain 16-bit state costs)
template <int F> struct Tile { float4 st[D * S / 4]; };                        // 32 KB
#ifndef HC_NOWAIT
#define HC_NOWAIT 0
#endif
#ifndef HC_TMA_SMALL
#define HC_TMA_SMALL 0
#endif
#ifndef HC_TMA
#define HC_TMA 1
#endif
// 16-bit tiles: TMA boxes (box hf = columns 64 hf .. + 63, row p at 128 p, 16B chunk c at c ^ (p & 7)); Q16 scales apart
template <> struct Tile<1> { uint4 q[D * S / 8]; };                            // 16 KB (+ 0.5 KB scales in Smem)
template <> struct Tile<2> { uint4 q[D * S / 8]; };                            // 16 KB
#ifndef HC_TMA_LD
#define HC_TMA_LD HC_TMA
#endif
#ifndef HC_TMA_ST
#define HC_TMA_ST HC_TMA
#endif
#ifndef HC_INPLACE
#define HC_INPLACE 1
#endif
template <int Q> constexpr bool TML = Q != 0 && HC_TMA_LD;
// in place (TMA loads): each box is read, overwritten with h_new by the same threads, stored by TMA, then reloaded with
// the next item's box, so no output buffer is needed (37 KB of shared memory -> 5-6 CTAs / SM)
template <int Q> constexpr bool IPL = TML<Q> && HC_INPLACE;
template <int Q> constexpr bool TMS = Q != 0 && HC_TMA_ST && !IPL<Q>;
template <int Q> constexpr bool TM = TML<Q> || TMS<Q>;
// threads issuing the state stores (warp 0 issues the loads): box 0 / box 1 + scales, one bulk group each in flight
constexpr int T_ST0 = 32, T_ST1 = 64;
struct Empty {};
#ifndef HC_DB
#define HC_DB 0
#endif
// 16-bit states: two tiles (2 x 16.5 KB still fits 4 CTAs / SM), so item k + 1 is fetched from the start of item k and
// no barrier separates the column halves
template <int Q> constexpr int NBUF = (Q != 0 && HC_DB) ? 2 : 1;
template <int Q> struct __align__(1024) Smem {
  Tile<Q> tile[NBUF<Q>];
  std::conditional_t<TMS<Q>, Tile<Q>, Empty> ob;   // TMA: h_new of the item, stored by two box writes
  alignas(16) float isc[(IPL<Q> ? 2 : 1) * (Q == 1 ? D * 2 : 4)]; alignas(16) float osc[Q == 1 && (TMS<Q> || IPL<Q>) ? D * 2 : 4];   // Q16 scales in / out [row][half]
  uint64_t bar[3];   // TMA: state boxes 0 / 1 (in place) or items k & 1; in place + HC_TMA_SMALL: bar[2] small data
  Small sm[2];
  float bias_k[R * S], bias_q[R * S];
  float xproj[R * D], zproj[R * D], outproj[R * D];
  float wb1[S], wc1[S];
  __nv_bfloat16 sBk[8][LDB];            // rows 0..3 B'[r], 4..7 previous B'[r] (permuted columns)
  __nv_bfloat16 sCb[4][LDB];            // rotated C'[r] (permuted columns)
  int slot[MAXK];
};

// PART 1: the pools (state, previous B', angles, previous x), PART 2: the in_proj output rows, 3: both
template <int Q, int PART = 3>
__device__ __forceinline__ void issue(const Args& a, Tile<Q>& tl, Small& sg, int b, int slot, int h, int t, float* isc) {
  const int srow = (slot >= 0 && slot < a.P) ? slot : 0;
  if constexpr (PART & 1 && !(TML<Q> && HC_TMA_SMALL == 0)) {
  if constexpr (TML<Q>) {}
  else if constexpr (Q) {
    const int16_t* src = reinterpret_cast<const int16_t*>(a.ssm) + (size_t)srow * a.ssm_sst + (size_t)h * D * S;
#pragma unroll
    for (int i = 0; i < 8; ++i) { const int c = t + i * THREADS, p = c >> 4, ch = c & 15; cp16(&tl.q[swz16(p, ch >> 1, ch & 1)], src + c * 8); }
    if constexpr (Q == 1) if (t < 32) cp16(isc + t * 4, a.ssc + (size_t)srow * a.ssc_sst + (size_t)h * D * 2 + t * 4);
  } else {
    const float* src = reinterpret_cast<const float*>(a.ssm) + (size_t)srow * a.ssm_sst + (size_t)h * D * S;
#pragma unroll
    for (int i = 0; i < 16; ++i) { const int c = t + i * THREADS, p = c >> 5, ch = c & 31; cp16(&tl.st[swz32(p, ch)], src + c * 4); }
  }
  }
  // 16B chunks: c = t (< 132: in_proj B / C / angle row) and c = t + 128 (row tail, head row, previous B', angles, x)
  const __nv_bfloat16* zr = a.zb + (size_t)b * a.zb_stride;
  if constexpr (PART & 2) cp16(sg.dyn + t * 8, zr + N1 + t * 8);
  const int c = t + THREADS;
  if (c < 150) {
    if constexpr (PART & 2) {
      if (c < 132) cp16(sg.dyn + c * 8, zr + N1 + c * 8);
      else cp16(sg.hrow + (c - 132) * 8, zr + ((h * NZX) & ~7) + (c - 132) * 8);
    }
  } else if constexpr (PART & 1) {
    if (c < 214) { const int q = c - 150, r = q >> 4, j = q & 15; cp16(sg.kprev + r * S + j * 8, a.kp + (size_t)srow * a.kp_sst + ((size_t)r * H + h) * S + j * 8); }
    else if (c < 222) cp16(sg.angst + (c - 214) * 4, a.angle + (size_t)srow * a.ang_sst + (size_t)h * NA + (c - 214) * 4);
    else if (c < 230) cp16(sg.xs + (c - 222) * 8, a.vp + (size_t)srow * a.vp_sst + (size_t)h * D + (c - 222) * 8);
  }
}

// TMA issue of item (b, slot) by warp 0: state boxes, Q16 scales, pools and in_proj rows on one mbarrier
template <int Q, int PART = 3>
__device__ __forceinline__ void issue_tm(const Args& a, const CUtensorMap* tm, Smem<Q>& S_, Small& sg, uint64_t* bar, int b, int slot, int h, int lane) {
  const int srow = (slot >= 0 && slot < a.P) ? slot : 0;
  if constexpr (PART & 1) {
    if (lane == 0) mbar_expect(bar, D * S * 2 + (Q == 1 ? D * 2 * 4 : 0) + (HC_TMA_SMALL ? (int)sizeof(Small) : 0));
    __syncwarp();
    if (lane < 2) tma_g2s(&S_.tile[0].q[lane * 512], tm, 64 * lane, h * D, srow, bar);
    else if (lane == 2) { if constexpr (Q == 1) bulk_g2s(S_.isc, a.ssc + (size_t)srow * a.ssc_sst + (size_t)h * D * 2, D * 2 * 4, bar); }
    else if (!HC_TMA_SMALL) {}
    else if (lane < 7) { const int r = lane - 3; bulk_g2s(sg.kprev + r * S, a.kp + (size_t)srow * a.kp_sst + ((size_t)r * H + h) * S, S * 2, bar); }
    else if (lane == 7) bulk_g2s(sg.angst, a.angle + (size_t)srow * a.ang_sst + (size_t)h * NA, NA * 4, bar);
    else if (lane == 8) bulk_g2s(sg.xs, a.vp + (size_t)srow * a.vp_sst + (size_t)h * D, D * 2, bar);
  }
  if constexpr (PART & 2 && HC_TMA_SMALL) {
    const __nv_bfloat16* zr = a.zb + (size_t)b * a.zb_stride;
    if (lane == 9) bulk_g2s(sg.dyn, zr + N1, NDYN * 2, bar);
    else if (lane == 10) bulk_g2s(sg.hrow, zr + ((h * NZX) & ~7), HROW * 2, bar);
  }
}

// in-place mode: one thread loads box `bx` of item (slot) on its own mbarrier (box 0 also brings the Q16 scales)
template <int Q>
__device__ __forceinline__ void issue_box(const Args& a, const CUtensorMap* tm, Smem<Q>& S_, int bx, int slot, int h, float* isc) {
  const int srow = (slot >= 0 && slot < a.P) ? slot : 0;
  uint64_t* bar = &S_.bar[bx];
  mbar_expect(bar, D * 64 * 2 + (Q == 1 && bx == 0 ? D * 2 * 4 : 0));
  tma_g2s(&S_.tile[0].q[bx * 512], tm, 64 * bx, h * D, srow, bar);
  if constexpr (Q == 1) if (bx == 0) bulk_g2s(isc, a.ssc + (size_t)srow * a.ssc_sst + (size_t)h * D * 2, D * 2 * 4, bar);
}

// in place + HC_TMA_SMALL: item b's in_proj rows and pool rows (previous B', angles, x) by bulk copies (warp 0 lanes)
template <int Q, int PART = 3>
__device__ __forceinline__ void issue_small_tm(const Args& a, Small& sg, uint64_t* bar, int b, int slot, int h, int lane) {
  const int srow = (slot >= 0 && slot < a.P) ? slot : 0;
  if constexpr (PART & 1) { if (lane == 0) mbar_expect(bar, (int)sizeof(Small)); __syncwarp(); }
  if constexpr (PART & 1) {
    if (lane < 4) bulk_g2s(sg.kprev + lane * S, a.kp + (size_t)srow * a.kp_sst + ((size_t)lane * H + h) * S, S * 2, bar);
    else if (lane == 4) bulk_g2s(sg.angst, a.angle + (size_t)srow * a.ang_sst + (size_t)h * NA, NA * 4, bar);
    else if (lane == 5) bulk_g2s(sg.xs, a.vp + (size_t)srow * a.vp_sst + (size_t)h * D, D * 2, bar);
  }
  if constexpr (PART & 2) {
    const __nv_bfloat16* zr = a.zb + (size_t)b * a.zb_stride;
    if (lane == 6) bulk_g2s(sg.dyn, zr + N1, NDYN * 2, bar);
    else if (lane == 7) bulk_g2s(sg.hrow, zr + ((h * NZX) & ~7), HROW * 2, bar);
  }
}
template <int Q> constexpr bool IPS = Q != 0 && HC_INPLACE && HC_TMA_LD && HC_TMA_SMALL;

// the 4 consecutive state columns 16 jp + 4 qd .. + 3 of row p
template <int Q> __device__ __forceinline__ int tmo(int p, int jl, int qd, int hf) { return hf * 512 + p * 8 + ((2 * jl + (qd >> 1)) ^ (p & 7)); }
template <int Q> __device__ __forceinline__ float4 tile_get(const Tile<Q>& tl, int p, int jp, int qd, int hf) {
  if constexpr (TML<Q>) {
    const uint2 w = reinterpret_cast<const uint2*>(&tl.q[tmo<Q>(p, jp & 3, qd, hf)])[qd & 1];
    if constexpr (Q == 1) { const float2 lo = i16x2(w.x), hi = i16x2(w.y); return make_float4(lo.x, lo.y, hi.x, hi.y); }
    else return make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u), __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u));
  } else if constexpr (Q == 1) {
    const uint2 w = reinterpret_cast<const uint2*>(&tl.q[swz16(p, jp, qd >> 1)])[qd & 1];   // raw mantissas
    const float2 lo = i16x2(w.x), hi = i16x2(w.y);
    return make_float4(lo.x, lo.y, hi.x, hi.y);
  } else if constexpr (Q == 2) {
    const uint2 w = reinterpret_cast<const uint2*>(&tl.q[swz16(p, jp, qd >> 1)])[qd & 1];
    return make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u), __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u));
  } else {
    return tl.st[swz32(p, 4 * jp + qd)];
  }
}

// CTAs / SM: fp32 4 (52 KB of shared memory each); 16-bit states in place 6 (37 KB, <= 80 registers, no spills)
#ifndef HC_MINB
#define HC_MINB 4
#endif
#ifndef HC_MINB16
#define HC_MINB16 (HC_INPLACE && HC_TMA_LD ? 6 : 4)
#endif
template <int Q>
__global__ void __launch_bounds__(THREADS, Q == 0 ? HC_MINB : HC_MINB16) mamba3_step_hc_kernel(const Args a, const __grid_constant__ CUtensorMap tmap) {
  extern __shared__ __align__(1024) char smem_raw[];
  Smem<Q>& S_ = *reinterpret_cast<Smem<Q>*>(smem_raw);   // dynamic shared memory starts 1 KB-aligned (checked for TMA)
  const CUtensorMap* tm = &tmap;
  // bf16 terms of the update operand (alpha-free gamma x / beta x_prev projections) and of the C dot operand (h_new).
  // fp32: 3 (bit-identical to mamba3_step_split). Q16: 2 update terms (16 mantissa bits, below the storage error:
  // 64 chained steps on real weights keep the state at the fp32 path's error vs fp64, 6e-5..1e-4; 1 term: 1e-3,
  // and no faster on B300 since the TMA / in-place rework), 1 C-dot term (feeds only the bf16 output y: y error
  // 2.02e-3 vs 1.93e-3 fp32 at a 1.58e-3 bf16 floor; 2 terms: equal to fp32, +17% time). bf16 state: 1 each.
#ifndef Q16_NS
#define Q16_NS 2
#endif
#ifndef Q16_NSC   // bf16 terms of h_new in the Q16 C dot
#define Q16_NSC 1
#endif
  constexpr int NS = Q == 0 ? 3 : Q == 1 ? Q16_NS : 1;
  constexpr int NSC = Q == 0 ? 3 : Q == 1 ? Q16_NSC : 1;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int qd = lane & 3, g = lane >> 2, p0 = warp * 16 + (TM<Q> ? ((g & 3) << 1) | (g >> 2) : g);
  const int mi = lane >> 3, li = lane & 7;
  const int h = blockIdx.x % H, nb = gridDim.x / H, b0 = blockIdx.x / H;
  if (b0 >= a.B) return;
  const int nk = (a.B - b0 + nb - 1) / nb;
  const int hoff = h * NZX - ((h * NZX) & ~7);
  // per-head constants (weights) and this CTA's slots
  for (int i = t; i < R * S; i += THREADS) {
    const int r = i / S, j = i % S;
    S_.bias_k[i] = a.bias_k[((size_t)r * H + h) * S + j]; S_.bias_q[i] = a.bias_q[((size_t)r * H + h) * S + j];
  }
  for (int i = t; i < R * D; i += THREADS) {
    const int r = i / D, p = i % D; const size_t off = ((size_t)r * H + h) * D + p;
    S_.xproj[i] = a.xproj[off]; S_.zproj[i] = a.zproj[off]; S_.outproj[i] = a.outproj[off];
  }
  for (int i = t; i < S; i += THREADS) { S_.wb1[i] = a.wb[i] + 1.f; S_.wc1[i] = a.wc[i] + 1.f; }
  for (int i = t; i < nk; i += THREADS) S_.slot[i] = a.slots[b0 + i * nb];
  const float dtb = a.dt_bias[h], Dh = a.Dp[h];
  if constexpr (TM<Q>) if (smem_u32(smem_raw) & 1023) __trap();
  // 16-bit states: this lane's B / C norm weights and biases (row = warp) stay in registers across items
  float rwb[4], rwc[4], rbk[4], rbq[4];
  if constexpr (Q != 0 && HC_REGW) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int j = lane + 32 * i;
      rwb[i] = a.wb[j] + 1.f; rwc[i] = a.wc[j] + 1.f;
      rbk[i] = a.bias_k[((size_t)warp * H + h) * S + j]; rbq[i] = a.bias_q[((size_t)warp * H + h) * S + j];
    }
  }
  if constexpr (TML<Q>) if (t == 0) { mbar_init(&S_.bar[0], 1); mbar_init(&S_.bar[1], 1); mbar_init(&S_.bar[2], 1); asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
  __syncthreads();
  // the pools are not the previous kernel's output
  if constexpr (IPL<Q>) { if (t == 0) { issue_box<Q>(a, tm, S_, 0, S_.slot[0], h, S_.isc); issue_box<Q>(a, tm, S_, 1, S_.slot[0], h, S_.isc); } }
  else if constexpr (TML<Q>) { if (warp == 0) issue_tm<Q, 1>(a, tm, S_, S_.sm[0], &S_.bar[0], b0, S_.slot[0], h, lane); }
  if constexpr (IPS<Q>) { if (warp == 0) issue_small_tm<Q, 1>(a, S_.sm[0], &S_.bar[2], b0, S_.slot[0], h, lane); }
  else if constexpr (!TML<Q> || !HC_TMA_SMALL) issue<Q, 1>(a, S_.tile[0], S_.sm[0], b0, S_.slot[0], h, t, S_.isc);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (a.pdl) asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  if constexpr (TML<Q> && !IPL<Q>) { if (warp == 0) issue_tm<Q, 2>(a, tm, S_, S_.sm[0], &S_.bar[0], b0, S_.slot[0], h, lane); }
  if constexpr (IPS<Q>) { if (warp == 0) issue_small_tm<Q, 2>(a, S_.sm[0], &S_.bar[2], b0, S_.slot[0], h, lane); }
  else if constexpr (!TML<Q> || !HC_TMA_SMALL) { issue<Q, 2>(a, S_.tile[0], S_.sm[0], b0, S_.slot[0], h, t, S_.isc); cp_commit(); }
  for (int k = 0; k < nk; ++k) {
    const int b = b0 + k * nb, slot = S_.slot[k];
    const bool valid = slot >= 0 && slot < a.P;
    Small& sg = S_.sm[k & 1];
    if constexpr (IPL<Q>) mbar_wait(&S_.bar[0], k & 1);   // box 0 (+ scales) of item k
    else if constexpr (TML<Q>) mbar_wait(&S_.bar[k & 1], (k >> 1) & 1);
    if constexpr (IPS<Q>) mbar_wait(&S_.bar[2], k & 1);
    if constexpr (!TML<Q> || !HC_TMA_SMALL) cp_wait_all();   // (IPS: no cp.async)
    __syncthreads();   // item k landed; every warp is done with item k - 1
    if constexpr (TMS<Q>) if (t == T_ST1 && k > 0) {   // second box (+ scales) of item k - 1
      const int ps = S_.slot[k - 1];
      if (ps >= 0 && ps < a.P) {
        tma_s2g(tm, 64, h * D, ps, &S_.ob.q[512]);
        if constexpr (Q == 1) bulk_s2g(a.ssc + (size_t)ps * a.ssc_sst + (size_t)h * D * 2, S_.osc, D * 2 * 4);
      }
      bulk_commit();
    }
    if constexpr (IPL<Q>) if (t == T_ST1 && k > 0) {   // store box 1 (+ scales) of item k - 1, then reload it with item k's
      const int ps = S_.slot[k - 1];
      if (ps >= 0 && ps < a.P) {
        tma_s2g(tm, 64, h * D, ps, &S_.tile[0].q[512]);
        if constexpr (Q == 1) bulk_s2g(a.ssc + (size_t)ps * a.ssc_sst + (size_t)h * D * 2, S_.osc, D * 2 * 4);
      }
      bulk_commit();
      bulk_wait_read<0>();
      issue_box<Q>(a, tm, S_, 1, slot, h, nullptr);
    }
    Tile<Q>& tk = S_.tile[k % NBUF<Q>];
    if constexpr (NBUF<Q> == 2) {   // the other tile / small buffer held item k - 1
      if (k + 1 < nk) issue<Q>(a, S_.tile[(k + 1) & 1], S_.sm[(k + 1) & 1], b0 + (k + 1) * nb, S_.slot[k + 1], h, t, S_.isc);
      cp_commit();
    }
    const __nv_bfloat16* hr = sg.hrow + hoff;
#ifdef HC_NOSCAL
    const float dt = bf2f(hr[2 * D]) + dtb, A = -bf2f(hr[2 * D + 1]), trap = bf2f(hr[2 * D + 2]);
#else
    constexpr bool FM = Q != 0 && HC_FASTMATH;
    const float dt = softplus_<FM>(bf2f(hr[2 * D]) + dtb);
    const float A = fminf(-softplus_<FM>(bf2f(hr[2 * D + 1])), -a.a_floor);
    const float trap = sigmoid_<FM>(bf2f(hr[2 * D + 2]));
#endif
    const float alpha = __expf(A * dt), gamma = trap * dt, beta = (1.f - trap) * dt * alpha;
    // ---- warp r: row r of B / C (as _bc_norm_kernel + the step's bias / rotation), columns lane + 32 i
    __nv_bfloat16 bn[4];
    float th;
    uint2 bnv;   // 16-bit states: B' columns 4 lane .. + 3 (natural order) for the k pool
    if constexpr (FM && HC_VEC) {   // vectorized: lane owns columns 4 lane .. 4 lane + 3 (slower: 2 x 8 shuffles, no wavefront saving)
      const int r = warp, c4 = 4 * lane;
      const uint2 bw = *reinterpret_cast<const uint2*>(&sg.dyn[r * S + c4]), cw = *reinterpret_cast<const uint2*>(&sg.dyn[R * S + r * S + c4]);
      float fb[4] = {__uint_as_float(bw.x << 16), __uint_as_float(bw.x & 0xffff0000u), __uint_as_float(bw.y << 16), __uint_as_float(bw.y & 0xffff0000u)};
      float fc[4] = {__uint_as_float(cw.x << 16), __uint_as_float(cw.x & 0xffff0000u), __uint_as_float(cw.y << 16), __uint_as_float(cw.y & 0xffff0000u)};
      float sb = (fb[0] * fb[0] + fb[1] * fb[1]) + (fb[2] * fb[2] + fb[3] * fb[3]), sc = (fc[0] * fc[0] + fc[1] * fc[1]) + (fc[2] * fc[2] + fc[3] * fc[3]);
#pragma unroll
      for (int o = 16; o; o >>= 1) { sb += __shfl_xor_sync(0xffffffffu, sb, o); sc += __shfl_xor_sync(0xffffffffu, sc, o); }
      const float ib = rsqrt_<FM>(sb / S + a.eps), ic = rsqrt_<FM>(sc / S + a.eps);
      th = sg.angst[lane] + tanh_<FM>(bf2f(sg.dyn[2 * R * S + lane])) * dt * 3.14159265358979323846f;
      th -= 6.283185307179586f * floorf(th / 6.283185307179586f);
      float sn, cs; sincos_<FM>(th, &sn, &cs);   // angle `lane`
      const float4 w1 = *reinterpret_cast<const float4*>(&S_.wb1[c4]), w2 = *reinterpret_cast<const float4*>(&S_.wc1[c4]);
      const float4 bk = *reinterpret_cast<const float4*>(&S_.bias_k[r * S + c4]), bq = *reinterpret_cast<const float4*>(&S_.bias_q[r * S + c4]);
      const float wbv[4] = {w1.x, w1.y, w1.z, w1.w}, wcv[4] = {w2.x, w2.y, w2.z, w2.w}, bkv[4] = {bk.x, bk.y, bk.z, bk.w}, bqv[4] = {bq.x, bq.y, bq.z, bq.w};
      float bv[4], cv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        bv[i] = bf2f(__float2bfloat16(fb[i] * ib * wbv[i])) + bkv[i];
        cv[i] = bf2f(__float2bfloat16(fc[i] * ic * wcv[i])) + bqv[i];
      }
      // columns j < 32 (lanes 0..7) and j + 64 (lanes 16..23) are the rotated pair of angle j
      const int jb = 4 * (lane & 7);
      const bool lo = lane < 8, hi = (lane & 24) == 16;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float ci = __shfl_sync(0xffffffffu, cs, jb + i), si = __shfl_sync(0xffffffffu, sn, jb + i);
        const float pb = __shfl_xor_sync(0xffffffffu, bv[i], 16), pc = __shfl_xor_sync(0xffffffffu, cv[i], 16);
        if (lo) { bv[i] = bv[i] * ci - pb * si; cv[i] = cv[i] * ci - pc * si; }
        else if (hi) { bv[i] = pb * si + bv[i] * ci; cv[i] = pc * si + cv[i] * ci; }
      }
      // permuted columns: 4 q + {0, 1, 2, 3} of 16-group G -> 16 G + {2 q, 2 q + 1, 2 q + 8, 2 q + 9}
      const int pc0 = (c4 & ~15) + 2 * (lane & 3);
      bnv = make_uint2(pack_bf16(bv[0], bv[1]), pack_bf16(bv[2], bv[3]));
      const uint2 kw = *reinterpret_cast<const uint2*>(&sg.kprev[r * S + c4]);
      *reinterpret_cast<uint32_t*>(&S_.sBk[r][pc0]) = bnv.x; *reinterpret_cast<uint32_t*>(&S_.sBk[r][pc0 + 8]) = bnv.y;
      *reinterpret_cast<uint32_t*>(&S_.sBk[4 + r][pc0]) = kw.x; *reinterpret_cast<uint32_t*>(&S_.sBk[4 + r][pc0 + 8]) = kw.y;
      *reinterpret_cast<uint32_t*>(&S_.sCb[r][pc0]) = pack_bf16(cv[0], cv[1]); *reinterpret_cast<uint32_t*>(&S_.sCb[r][pc0 + 8]) = pack_bf16(cv[2], cv[3]);
    } else {
      const int r = warp;
      float fb[4], fc[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) { fb[i] = bf2f(sg.dyn[r * S + lane + 32 * i]); fc[i] = bf2f(sg.dyn[R * S + r * S + lane + 32 * i]); }
      float sb = 0.f, sc = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) { sb += fb[i] * fb[i]; sc += fc[i] * fc[i]; }
#pragma unroll
      for (int o = 16; o; o >>= 1) { sb += __shfl_xor_sync(0xffffffffu, sb, o); sc += __shfl_xor_sync(0xffffffffu, sc, o); }
      const float ib = rsqrt_<FM>(sb / S + a.eps), ic = rsqrt_<FM>(sc / S + a.eps);
      th = sg.angst[lane] + tanh_<FM>(bf2f(sg.dyn[2 * R * S + lane])) * dt * 3.14159265358979323846f;
      th -= 6.283185307179586f * floorf(th / 6.283185307179586f);
      float sn, cs; sincos_<FM>(th, &sn, &cs);
      float bv[4], cv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if constexpr (Q != 0 && HC_REGW) {
          bv[i] = bf2f(__float2bfloat16(fb[i] * ib * rwb[i])) + rbk[i];
          cv[i] = bf2f(__float2bfloat16(fc[i] * ic * rwc[i])) + rbq[i];
        } else {
          bv[i] = bf2f(__float2bfloat16(fb[i] * ib * S_.wb1[lane + 32 * i])) + S_.bias_k[r * S + lane + 32 * i];
          cv[i] = bf2f(__float2bfloat16(fc[i] * ic * S_.wc1[lane + 32 * i])) + S_.bias_q[r * S + lane + 32 * i];
        }
      }
      {   // columns (lane, lane + 64) are the rotated pair of angle `lane`
        const float b0_ = bv[0] * cs - bv[2] * sn, b1 = bv[0] * sn + bv[2] * cs, c0 = cv[0] * cs - cv[2] * sn, c1 = cv[0] * sn + cv[2] * cs;
        bv[0] = b0_; bv[2] = b1; cv[0] = c0; cv[2] = c1;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int c = perm(lane + 32 * i);
        bn[i] = __float2bfloat16(bv[i]);
        S_.sBk[r][c] = bn[i]; S_.sBk[4 + r][c] = sg.kprev[r * S + lane + 32 * i]; S_.sCb[r][c] = __float2bfloat16(cv[i]);
      }
    }
    if constexpr (TMS<Q>) if (t == T_ST0) bulk_wait_read<0>();   // first box of item k - 1 read: ob box 0 is free
    __syncthreads();   // B' / C' ready
    if (valid) {
#pragma unroll
      if constexpr (FM && HC_VEC) *reinterpret_cast<uint2*>(&a.kp[(size_t)slot * a.kp_sst + ((size_t)warp * H + h) * S + 4 * lane]) = bnv;
      else for (int i = 0; i < 4; ++i) a.kp[(size_t)slot * a.kp_sst + ((size_t)warp * H + h) * S + lane + 32 * i] = bn[i];
      if (warp == 0) a.angle[(size_t)slot * a.ang_sst + (size_t)h * NA + lane] = th;
    }
    float xp[2], zp[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) { zp[e] = bf2f(hr[p0 + 8 * e]); xp[e] = bf2f(hr[D + p0 + 8 * e]); }
    uint32_t ah[3][4];
    float av[2][2];
    {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float xs = bf2f(sg.xs[p0 + 8 * e]);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int kk = 2 * qd + jj;
          const float xr = S_.xproj[(kk & 3) * D + p0 + 8 * e];
          av[e][jj] = (kk < 4) ? gamma * xp[e] * xr : beta * xs * xr;
        }
      }
#pragma unroll
      for (int e = 0; e < 2; ++e) splitn_x2<NS>(av[e][0], av[e][1], &ah[0][e], &ah[1][e], &ah[2][e]);
#pragma unroll
      for (int q = 0; q < 3; ++q) { ah[q][2] = 0u; ah[q][3] = 0u; }
    }
    // C dot accumulators: fp32 one per column group (order of mamba3_step_split), 16-bit states one chain
    constexpr int NOC = Q == 0 ? 4 : 1;
    float oc[NOC][4];
#pragma unroll
    for (int q = 0; q < NOC; ++q) { oc[q][0] = 0.f; oc[q][1] = 0.f; oc[q][2] = 0.f; oc[q][3] = 0.f; }
    if (valid && qd == 0) {
      a.vp[(size_t)slot * a.vp_sst + (size_t)h * D + p0] = __float2bfloat16(xp[0]);
      a.vp[(size_t)slot * a.vp_sst + (size_t)h * D + p0 + 8] = __float2bfloat16(xp[1]);
    }
    float fs0[2] = {1.f, 1.f};
    float sc_h0[2] = {1.f, 1.f};   // Q16: half-0 scales, stored with half 1's   // Q16: alpha scale of the rows' current half (oc is in its units)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {   // column halves: 32 accumulator registers
      float acc[8][4], fs[2], rs[2] = {1.f, 1.f};
      if constexpr (IPL<Q>) if (hf == 1) {   // box 0 overwritten by every warp: store it, reload it with item k + 1
        fence_async_smem();
        __syncthreads();
        const int kn = k + 1;
        if (t == T_ST0) {
          if (valid) tma_s2g(tm, 0, h * D, slot, &S_.tile[0].q[0]);
          bulk_commit();
          bulk_wait_read<0>();
          if (kn < nk) issue_box<Q>(a, tm, S_, 0, S_.slot[kn], h, S_.isc + (kn & 1) * D * 2);
        }
        if constexpr (IPS<Q>) { if (warp == 0 && kn < nk) issue_small_tm<Q>(a, S_.sm[kn & 1], &S_.bar[2], b0 + kn * nb, S_.slot[kn], h, lane); }
        else { if (kn < nk) issue<Q>(a, S_.tile[0], S_.sm[kn & 1], b0 + kn * nb, S_.slot[kn], h, t, S_.isc); cp_commit(); }
        mbar_wait(&S_.bar[1], k & 1);   // box 1 of item k
      }
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        if constexpr (Q == 1) {
          // q units = alpha * scale, floored at QU_MIN: below it (old state fully decayed, e.g. A dt < -34 at scale 1e-5) the
          // update divided by the unit would overflow the accumulators (inf / NaN state). Floored rows seed the
          // accumulators with q * rs (rs = alpha * scale / QU_MIN < 1): same value, no overflow.
          const float f = alpha * S_.isc[(IPL<Q> ? (k & 1) * D * 2 : 0) + (p0 + 8 * e) * 2 + hf];
          const bool lo = !(f >= QU_MIN);
          fs[e] = lo ? QU_MIN : f;
          rs[e] = lo ? f * (1.f / QU_MIN) : 1.f;
        } else {
          fs[e] = alpha;
        }
      }
      // warp-uniform: only warps holding a floored row take the multiplying seed
      const bool seed_mul = Q == 1 && __any_sync(0xffffffffu, rs[0] != 1.f || rs[1] != 1.f);
      if constexpr (Q == 1) {
        // Q16: the accumulators hold h_new / (alpha scale) of their row and half ("q units"): the state tile is consumed
        // as is, the update operand is divided by alpha scale (2 values per thread), the C dot result is rescaled
        // between halves, and the requantization needs no scale multiply
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float ir; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(ir) : "f"(fs[e]));
          splitn_x2<NS>(av[e][0] * ir, av[e][1] * ir, &ah[0][e], &ah[1][e], &ah[2][e]);
        }
        if (hf == 1) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float ir; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(ir) : "f"(fs[e]));
            const float rt = fs0[e] * ir;
#pragma unroll
            for (int q = 0; q < NOC; ++q) { oc[q][2 * e] *= rt; oc[q][2 * e + 1] *= rt; }
          }
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) fs0[e] = fs[e];
      }
#pragma unroll
      for (int jl = 0; jl < 4; ++jl)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float4 v = tile_get<Q>(tk, p0 + 8 * e, 4 * hf + jl, qd, hf);
          if constexpr (Q == 1) {
            if (seed_mul) {
              acc[2 * jl][2 * e] = rs[e] * v.x; acc[2 * jl][2 * e + 1] = rs[e] * v.y;
              acc[2 * jl + 1][2 * e] = rs[e] * v.z; acc[2 * jl + 1][2 * e + 1] = rs[e] * v.w;
            } else {
              acc[2 * jl][2 * e] = v.x; acc[2 * jl][2 * e + 1] = v.y; acc[2 * jl + 1][2 * e] = v.z; acc[2 * jl + 1][2 * e + 1] = v.w;
            }
          } else {
            acc[2 * jl][2 * e] = fs[e] * v.x; acc[2 * jl][2 * e + 1] = fs[e] * v.y;
            acc[2 * jl + 1][2 * e] = fs[e] * v.z; acc[2 * jl + 1][2 * e + 1] = fs[e] * v.w;
          }
        }
      if (NBUF<Q> == 1 && hf == 1 && !IPL<Q>) {   // tile consumed by every warp: fetch the next item
        if constexpr (TMS<Q>) { fence_async_smem(); if (t == T_ST1) bulk_wait_read<0>(); }   // ob box 0 written; box 1 of item k - 1 read
        __syncthreads();
        const int kn = k + 1;
        if constexpr (TM<Q>) {
          if constexpr (TMS<Q>) if (t == T_ST0) { if (valid) tma_s2g(tm, 0, h * D, slot, &S_.ob.q[0]); bulk_commit(); }
          if (warp == 0) {
            if (TML<Q> && kn < nk) issue_tm<Q>(a, tm, S_, S_.sm[kn & 1], &S_.bar[kn & 1], b0 + kn * nb, S_.slot[kn], h, lane);
          }
        }
        if constexpr (!TML<Q> || !HC_TMA_SMALL) {
          if (kn < nk) issue<Q>(a, S_.tile[0], S_.sm[kn & 1], b0 + kn * nb, S_.slot[kn], h, t, S_.isc);
          cp_commit();
        }
      }
#pragma unroll
      for (int jl = 0; jl < 4; ++jl) {
        const int jp = 4 * hf + jl;
        // k rows 0..7 of the two n tiles; k rows 8..15 meet zero A columns, so the same (finite) registers serve
        uint32_t bfr[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(bfr[0]), "=r"(bfr[2]) : "r"(smem_u32(&S_.sBk[li][jp * 16 + (mi & 1) * 8])));
        if constexpr (Q == 0) {   // m16n8k16 with zero A columns 8..15 (bit-identical to mamba3_step_split)
          bfr[1] = bfr[0]; bfr[3] = bfr[2];
#pragma unroll
          for (int q = NS - 1; q >= 0; --q) { mma16816(acc[2 * jl], ah[q], bfr); mma16816(acc[2 * jl + 1], ah[q], bfr + 2); }
        } else {                  // K = 8 exactly (B' rows | previous B' rows)
#pragma unroll
          for (int q = NS - 1; q >= 0; --q) { mma1688(acc[2 * jl], ah[q], bfr[0]); mma1688(acc[2 * jl + 1], ah[q], bfr[2]); }
        }
      }
#pragma unroll
      for (int jl = 0; jl < 4; ++jl) {
        const int c = 4 * hf + jl;
        uint32_t bfr[2];   // n rows 4..7 alias rows 0..3 (their outputs are unused)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(bfr[0]), "=r"(bfr[1]) : "r"(smem_u32(&S_.sCb[li & 3][c * 16 + (mi & 1) * 8])));
        const float* v[4] = {&acc[2 * jl][0], &acc[2 * jl][2], &acc[2 * jl + 1][0], &acc[2 * jl + 1][2]};
        uint32_t a3[3][4];
#pragma unroll
        for (int q = 0; q < 4; ++q) splitn_x2<NSC>(v[q][0], v[q][1], &a3[0][q], &a3[1][q], &a3[2][q]);
#pragma unroll
        for (int q = NSC - 1; q >= 0; --q) mma16816(oc[jl % NOC], a3[q], bfr);
      }
#ifdef HC_NOST
      if (valid && a.pdl == 7) {
#else
      if (valid) {
#endif
        if constexpr (Q == 2) {
          uint16_t* dst = reinterpret_cast<uint16_t*>(a.ssm) + (size_t)slot * a.ssm_sst + ((size_t)h * D + p0) * S + 4 * qd;
#pragma unroll
          for (int e = 0; e < 2; ++e)
#pragma unroll
            for (int jl = 0; jl < 4; ++jl)
            {
              const uint2 w = make_uint2(pack_bf16(acc[2 * jl][2 * e], acc[2 * jl][2 * e + 1]), pack_bf16(acc[2 * jl + 1][2 * e], acc[2 * jl + 1][2 * e + 1]));
              if constexpr (IPL<Q>) reinterpret_cast<uint2*>(&tk.q[tmo<Q>(p0 + 8 * e, jl, qd, hf)])[qd & 1] = w;
              else if constexpr (TMS<Q>) reinterpret_cast<uint2*>(&S_.ob.q[tmo<Q>(p0 + 8 * e, jl, qd, hf)])[qd & 1] = w;
              else __stcs(reinterpret_cast<uint2*>(dst + e * 8 * S + 16 * (4 * hf + jl)), w);
            }
        } else if constexpr (Q == 1) {
          int16_t* dst = reinterpret_cast<int16_t*>(a.ssm) + (size_t)slot * a.ssm_sst + ((size_t)h * D + p0) * S + 4 * qd;
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float m = 0.f;
#pragma unroll
            for (int jl = 0; jl < 4; ++jl)
              m = fmaxf(m, fmaxf(fmaxf(fabsf(acc[2 * jl][2 * e]), fabsf(acc[2 * jl][2 * e + 1])),
                                 fmaxf(fabsf(acc[2 * jl + 1][2 * e]), fabsf(acc[2 * jl + 1][2 * e + 1]))));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1)); m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            // any (q, scale) with |q| <= 32767 decodes the same way: approximate reciprocal (|h| * inv <= 32767 (1 + 2^-21),
            // rounds to <= 32767) and q = rint(h * inv) in one FFMA (the correctly rounded quotient)
            float sc = m * 3.0518509475997192e-05f;
            if (!(sc >= 1.17549435e-38f)) sc = 1.f;   // zero (or a sub-FLT_MIN half: rcp.ftz would give inf) -> q = 0
            float inv; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(sc));
            auto qz2 = [&](float v0, float v1) {
              const float2 r = ffma2(make_float2(v0, v1), make_float2(inv, inv), make_float2(12615680.f, 12615680.f));
              return __byte_perm(__float_as_uint(r.x), __float_as_uint(r.y), 0x5410); };
#pragma unroll
            for (int jl = 0; jl < 4; ++jl) {
              const uint2 w = make_uint2(qz2(acc[2 * jl][2 * e], acc[2 * jl][2 * e + 1]), qz2(acc[2 * jl + 1][2 * e], acc[2 * jl + 1][2 * e + 1]));
              if constexpr (IPL<Q>) reinterpret_cast<uint2*>(&tk.q[tmo<Q>(p0 + 8 * e, jl, qd, hf)])[qd & 1] = w;
              else if constexpr (TMS<Q>) reinterpret_cast<uint2*>(&S_.ob.q[tmo<Q>(p0 + 8 * e, jl, qd, hf)])[qd & 1] = w;
              else __stcs(reinterpret_cast<uint2*>(dst + e * 8 * S + 16 * (4 * hf + jl)), w);
            }
            // both half scales of a row in one 8-byte store once the second half is done (contiguous across the quad rows)
            const float sv = m == 0.f ? 1.f : sc * fs[e];
            if (hf == 0) sc_h0[e] = sv;
            if constexpr (TMS<Q> || IPL<Q>) { if (hf == 1 && qd == 0) *reinterpret_cast<float2*>(&S_.osc[(p0 + 8 * e) * 2]) = make_float2(sc_h0[e], sv); }
            else if (hf == 1 && qd == 0) *reinterpret_cast<float2*>(a.ssc + (size_t)slot * a.ssc_sst + ((size_t)h * D + p0 + 8 * e) * 2) = make_float2(sc_h0[e], sv);
          }
        } else {
          float* dst = reinterpret_cast<float*>(a.ssm) + (size_t)slot * a.ssm_sst + ((size_t)h * D + p0) * S + 4 * qd;
#pragma unroll
          for (int e = 0; e < 2; ++e)
#pragma unroll
            for (int jl = 0; jl < 4; ++jl)
              __stcs(reinterpret_cast<float4*>(dst + e * 8 * S + 16 * (4 * hf + jl)),
                     make_float4(acc[2 * jl][2 * e], acc[2 * jl][2 * e + 1], acc[2 * jl + 1][2 * e], acc[2 * jl + 1][2 * e + 1]));
        }
      }
    }
    if constexpr (Q == 1) {
#pragma unroll
      for (int e = 0; e < 2; ++e)
#pragma unroll
        for (int q = 0; q < NOC; ++q) { oc[q][2 * e] *= fs0[e]; oc[q][2 * e + 1] *= fs0[e]; }
    }
    float ys[2] = {0.f, 0.f};
    if (qd < 2) {
#pragma unroll
      for (int e = 0; e < 2; ++e)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int r = 2 * qd + j, p = p0 + 8 * e, n = 2 * e + j;
          float orr = (NOC == 4 ? (oc[0][n] + oc[1 % NOC][n]) + (oc[2 % NOC][n] + oc[3 % NOC][n]) : oc[0][n]) + Dh * xp[e] * S_.xproj[r * D + p];
          const float zv = zp[e] * S_.zproj[r * D + p];
          orr *= __fdividef(zv, 1.f + __expf(-zv));
          ys[e] += orr * S_.outproj[r * D + p];
        }
    }
    ys[0] += __shfl_xor_sync(0xffffffffu, ys[0], 1); ys[1] += __shfl_xor_sync(0xffffffffu, ys[1], 1);
    if (qd == 0) {
      a.y[(size_t)b * a.y_stride + (size_t)h * D + p0] = __float2bfloat16(valid ? ys[0] : 0.f);
      a.y[(size_t)b * a.y_stride + (size_t)h * D + p0 + 8] = __float2bfloat16(valid ? ys[1] : 0.f);
    }
    if constexpr (TMS<Q> || IPL<Q>) fence_async_smem();   // box 1 / scales written (stored after the next barrier)
  }
  if constexpr (IPL<Q>) {
    __syncthreads();
    if (t == T_ST0) bulk_wait<0>();
    if (t == T_ST1) {
      const int ps = S_.slot[nk - 1];
      if (ps >= 0 && ps < a.P) {
        tma_s2g(tm, 64, h * D, ps, &S_.tile[0].q[512]);
        if constexpr (Q == 1) bulk_s2g(a.ssc + (size_t)ps * a.ssc_sst + (size_t)h * D * 2, S_.osc, D * 2 * 4);
      }
      bulk_commit();
      bulk_wait<0>();
    }
  }
  if constexpr (TMS<Q>) {
    __syncthreads();
    if (t == T_ST0) bulk_wait<0>();
    if (t == T_ST1) {
      const int ps = S_.slot[nk - 1];
      if (ps >= 0 && ps < a.P) {
        tma_s2g(tm, 64, h * D, ps, &S_.ob.q[512]);
        if constexpr (Q == 1) bulk_s2g(a.ssc + (size_t)ps * a.ssc_sst + (size_t)h * D * 2, S_.osc, D * 2 * 4);
      }
      bulk_commit();
      bulk_wait<0>();
    }
  }
}
}  // namespace

void mamba3_step_hc(torch::Tensor zb, torch::Tensor ssm_pool, std::optional<torch::Tensor> ssm_scale, torch::Tensor k_pool,
                    torch::Tensor v_pool, torch::Tensor angle_pool, torch::Tensor slots, torch::Tensor dt_bias, double a_floor,
                    torch::Tensor wb, torch::Tensor wc, double eps, torch::Tensor Dp, torch::Tensor xproj, torch::Tensor zproj,
                    torch::Tensor outproj, torch::Tensor bias_q, torch::Tensor bias_k, torch::Tensor y, int64_t ctas_per_sm, bool pdl) {
  const bool q16 = ssm_pool.dtype() == torch::kInt16, b16 = ssm_pool.dtype() == torch::kBFloat16;
  Args a;
  a.zb = reinterpret_cast<const __nv_bfloat16*>(zb.data_ptr()); a.zb_stride = zb.stride(0); a.B = zb.size(0);
  a.ssm = ssm_pool.data_ptr(); a.kp = reinterpret_cast<__nv_bfloat16*>(k_pool.data_ptr()); a.vp = reinterpret_cast<__nv_bfloat16*>(v_pool.data_ptr());
  a.angle = angle_pool.data_ptr<float>(); a.P = ssm_pool.size(0);
  a.ssm_sst = ssm_pool.stride(0); a.kp_sst = k_pool.stride(0); a.vp_sst = v_pool.stride(0); a.ang_sst = angle_pool.stride(0);
  a.ssc = nullptr; a.ssc_sst = 0;
  a.slots = slots.data_ptr<int>();
  a.dt_bias = dt_bias.data_ptr<float>(); a.a_floor = (float)a_floor; a.wb = wb.data_ptr<float>(); a.wc = wc.data_ptr<float>(); a.eps = (float)eps;
  a.Dp = Dp.data_ptr<float>(); a.xproj = xproj.data_ptr<float>(); a.zproj = zproj.data_ptr<float>(); a.outproj = outproj.data_ptr<float>();
  a.bias_q = bias_q.data_ptr<float>(); a.bias_k = bias_k.data_ptr<float>();
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr()); a.y_stride = y.stride(0); a.pdl = pdl ? 1 : 0;
  TORCH_CHECK(a.B >= 1 && zb.size(1) == N1 + NDYN && zb.stride(1) == 1 && zb.dtype() == torch::kBFloat16);
  TORCH_CHECK((zb.stride(0) * 2) % 16 == 0 && reinterpret_cast<uintptr_t>(zb.data_ptr()) % 16 == 0, "zb rows must be 16B aligned");
  TORCH_CHECK((q16 || b16 || ssm_pool.dtype() == torch::kFloat32) && ssm_pool.stride(1) == D * S && ssm_pool.stride(2) == S && ssm_pool.stride(3) == 1);
  if (q16) {
    TORCH_CHECK(ssm_scale.has_value() && ssm_scale->dtype() == torch::kFloat32 && ssm_scale->size(0) == a.P && ssm_scale->stride(1) == D * 2 &&
                ssm_scale->stride(2) == 2 && ssm_scale->stride(3) == 1 && (ssm_scale->stride(0) * 4) % 16 == 0, "Q16 needs scales [P, H, D, 2] fp32");
    a.ssc = ssm_scale->data_ptr<float>(); a.ssc_sst = ssm_scale->stride(0);
  }
  TORCH_CHECK(k_pool.stride(1) == H * S && k_pool.stride(2) == S && v_pool.stride(1) == D && angle_pool.stride(1) == NA);
  TORCH_CHECK((a.ssm_sst * ssm_pool.element_size()) % 16 == 0 && (a.kp_sst * 2) % 16 == 0 && (a.vp_sst * 2) % 16 == 0 && (a.ang_sst * 4) % 16 == 0, "pool slot strides must be 16B aligned");
  TORCH_CHECK(wb.dtype() == torch::kFloat32 && wc.dtype() == torch::kFloat32 && bias_q.dtype() == torch::kFloat32 && xproj.dtype() == torch::kFloat32);
  CUtensorMap tmap{};
  if ((q16 || b16) && HC_TMA) {   // [P, H * D, S] 16-bit view, 64 x 64 boxes, 128B swizzle (cached per pool)
    static std::unordered_map<const void*, std::pair<long long, CUtensorMap>> cache;
    const long long key = (long long)a.P * 1000003LL + a.ssm_sst;
    auto it = cache.find(a.ssm);
    if (it != cache.end() && it->second.first == key) tmap = it->second.second;
    else {
      static PFN_cuTensorMapEncodeTiled_v12000 enc = nullptr;
      if (!enc) {
        cudaDriverEntryPointQueryResult qr;
        TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&enc, cudaEnableDefault, &qr) == cudaSuccess && enc);
      }
      cuuint64_t dims[3] = {(cuuint64_t)S, (cuuint64_t)H * D, (cuuint64_t)a.P};
      cuuint64_t strides[2] = {(cuuint64_t)S * 2, (cuuint64_t)a.ssm_sst * 2};
      cuuint32_t box[3] = {64, (cuuint32_t)D, 1}, es[3] = {1, 1, 1};
      const CUresult r = enc(&tmap, CU_TENSOR_MAP_DATA_TYPE_UINT16, 3, a.ssm, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
      cache[a.ssm] = {key, tmap};
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(a.ssm) % 16 == 0 && (!q16 || reinterpret_cast<uintptr_t>(a.ssc) % 16 == 0));
  }
  auto launch = [&](auto kernel, int smem) {
    static const void* done[3] = {};
    bool seen = false;
    for (auto& d : done) { if (d == (const void*)kernel) { seen = true; break; } if (!d) { d = (const void*)kernel; break; } }
    if (!seen) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int nsm = 148; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, zb.get_device());
    int fit = 1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, kernel, THREADS, smem);
    const int cps = ctas_per_sm > 0 ? std::max(1, std::min<int>((int)ctas_per_sm, fit)) : fit;   // 0: as many as fit
    if (getenv("HC_DBG")) { static int once = 0; if (!once++) printf("hc: smem %d fit %d cps %d\n", smem, fit, cps); }
    int nb = std::max(1, std::min(a.B, nsm * cps / H));
    nb = std::max(nb, (a.B + MAXK - 1) / MAXK);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(nb * H); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = smem; cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = pdl ? 1 : 0;
    cudaLaunchKernelEx(&cfg, kernel, a, tmap);
  };
  if (q16) launch(mamba3_step_hc_kernel<1>, (int)sizeof(Smem<1>));
  else if (b16) launch(mamba3_step_hc_kernel<2>, (int)sizeof(Smem<2>));
  else launch(mamba3_step_hc_kernel<0>, (int)sizeof(Smem<0>));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step", &mamba3_step_hc, py::arg("zb"), py::arg("ssm"), py::arg("ssm_scale"), py::arg("kp"), py::arg("vp"), py::arg("angle"),
        py::arg("slots"), py::arg("dt_bias"), py::arg("a_floor"), py::arg("wb"), py::arg("wc"), py::arg("eps"), py::arg("Dp"),
        py::arg("xproj"), py::arg("zproj"), py::arg("outproj"), py::arg("bias_q"), py::arg("bias_k"), py::arg("y"),
        py::arg("ctas_per_sm") = 0, py::arg("pdl") = false);
}
