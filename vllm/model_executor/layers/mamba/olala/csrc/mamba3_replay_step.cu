// Olala Mamba-3 MIMO decode step with ReplaySSM (Liou et al., "ReplaySSM: Cache SSM Inputs, Not State",
// arXiv 2610.13778), adapted to the trapezoidal, rotary, rank-R Mamba-3 update. Built JIT by mamba3_step_cuda.py.
//
// The state of a request is a checkpoint S_c (the "anchor", fp32 or bf16, plus an optional fp8 read copy) and a
// ring of the tokens decoded since. Token i contributes  X_i (x) B'_i  (X_i[p][r] = x_i[p] xproj[r][p], B'_i the
// biased + rotated B) with weight gamma_i at its own step and beta_{i+1} at the next one, so after the current
// token (index n-1; index 0 is the token before the window, kept in k_pool / v_pool):
//     S = exp(L_{n-1}) S_c + sum_k w_k X_k (x) B'_k,  w_k = gamma_k e^{L_{n-1}-L_k} + beta_{k+1} e^{L_{n-1}-L_{k+1}}
// with L the cumulative log-decay A*dt over the window. Most steps only read S_c (preferably the fp8 copy) and
// append the token to the ring; the output is o = e^L S_c C'^T + sum_k w_k X_k G_k with G_k = B'_k C'^T.
// When the ring is full the step "flushes": S is accumulated in fp32 from the anchor and the ring and written back
// to the anchor and, independently, to the read copy. The read copy is never fed back into the recurrence, so its
// rounding error does not propagate; the anchor is rounded at most once per window (stochastically when bf16).
//
// Ring layout per slot: x (H, L, D) bf16; B' = rot(B + bias_k) per head (H, L, R, S) bf16, stored by the step that
// decoded the token (each B' is rotated once, at its own step, and only read afterwards); (A*dt, gamma, beta, -)
// (H, L, 4) fp32; (count, flush target) (H, 2) int32.
//
// Execution (organized around latency: the work per (token, head) is small and was a serial chain of loads,
// barriers and a single-thread scan): one CTA per (head, group of tokens) walks its tokens, with the head's
// constants (rotary biases, x/z/out projections) loaded once; a two-stage cp.async pipeline lands the next token's
// ring rows, z row and phase while the current token computes, and the per-token scalars (slot, ring count, dt, A,
// trap) are fetched two tokens ahead so a prefetch knows the ring extent. The fp8 checkpoint tile goes straight to
// registers (no shared memory, which bounds the CTAs per SM), with the mma k order permuted consistently between
// the tile and C'. Warp w owns
// state rows [16w, 16w + 16) in the mma fragment layout (lane: g = lane / 4 -> rows 16w + g and 16w + g + 8,
// qd = lane % 4 -> columns 8j + 2qd, +1). Tensor cores: G = B' C'^T (bf16), the fp8 checkpoint read S_c C'^T
// (fp8 -> f16 is exact, f16 C' is exact), the window term sum_k (w_k x_k) G_k and the flush update
// sum_k w_k X_k (x) B'_k, the last two with their fp32 operands split in two bf16 terms (hi + lo: ~fp32 accurate).
// B300, L = 8, bf16 anchor + fp8 read: 134 us per layer at 256 tokens, 495 at 1024 (the previous one-CTA-per-
// (head, token) kernel: 175 / 675; the plain fp32-state step: 161 / 604).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <algorithm>

namespace {
constexpr int D = 64, S = 128, R = 4, NA = 32, THREADS = 128, HALF = S / 2;
constexpr int LDB = S + 8;    // bf16 / f16 operand rows: conflict-free ldmatrix
constexpr float PI = 3.14159265358979323846f, TWO_PI = 6.283185307179586f, FP8_MAX = 448.f;

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(s)), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t v) {
  const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
  __half2 h = __floats2half2_rn(f.x, f.y); return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint16_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t hash32(uint32_t x) {   // lowbias32
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
// fp32 -> bf16 with stochastic rounding (add 16 random bits below the kept mantissa, truncate)
__device__ __forceinline__ __nv_bfloat16 bf16_sr(float v, uint32_t rnd) {
  uint32_t u = __float_as_uint(v);
  if ((u & 0x7f800000u) != 0x7f800000u) u = (u + (rnd & 0xffffu)) & 0xffff0000u;
  return __ushort_as_bfloat16((unsigned short)(u >> 16));
}

struct Args {
  float* angle; void* anchor; __nv_fp8_e4m3* rd; float* rd_scale; __nv_bfloat16* kp; __nv_bfloat16* vp;
  __nv_bfloat16* xr; __nv_bfloat16* br; float* scr; int* cnt;
  long long ang_sst, anc_sst, rd_sst, rds_sst, kp_sst, vp_sst, xr_sst, br_sst, scr_sst, cnt_sst;
  int P, H, L, N, tpc;   // tpc: tokens per CTA (upper bound)
  const float* A; const __nv_bfloat16* Bm; const __nv_bfloat16* Cm; const float* Dp; const __nv_bfloat16* x;
  const float* dt; const float* trap; const float* xproj; const float* zproj; const float* outproj;
  const __nv_bfloat16* z; const float* bias_q; const float* bias_k; const __nv_bfloat16* angp; const int* slots;
  __nv_bfloat16* y; long long bc_bstride, angp_stride;
};

// per-token buffers (two of them): everything a step reads, filled by cp.async
struct StageLayout {
  int bk, cb, xb, scr, z, angst, angp, total, bk_rows;
  __host__ __device__ StageLayout(int L) {
    const int n = L + 1; int o = 0;
    bk_rows = ((n * R + 15) / 16) * 16;
    bk = o;    o += bk_rows * LDB * 2;  // B' rows (k, r): token 0 = k_pool, ring, current (rotated in place)
    cb = o;    o += 8 * LDB * 2;        // C rows; 4..7 zero
    xb = o;    o += n * D * 2;          // x_k
    scr = o;   o += L * 4 * 4;          // ring (A*dt, gamma, beta, -)
    z = o;     o += D * 2;
    angst = o; o += NA * 4;             // carried phase
    angp = o;  o += 64;                 // phase projection (NA bf16)
    total = (o + 127) / 128 * 128;
  }
};
struct Layout {
  StageLayout st;
  int stage, bias, xproj, zproj, outproj, ch, G, w, slots, total;
  __host__ __device__ Layout(int L, int tpc) : st(L) {
    const int n = L + 1; int o = 0;
    stage = o;   o += 2 * st.total;
    bias = o;    o += 2 * R * S * 4;    // bias_k, bias_q of this head
    xproj = o;   o += R * D * 4;
    zproj = o;   o += R * D * 4;
    outproj = o; o += R * D * 4;
    ch = o;      o += 8 * LDB * 2;      // C' f16, rows 4..7 zero
    G = o;       o += n * R * R * 4;
    w = o;       o += n * 4;
    slots = o;   o += tpc * 4;
    total = (o + 15) / 16 * 16;
  }
};

struct Tok {   // per-token scalars
  int slot, hc, lim; float dt, A, trap; bool valid;
};

__device__ __forceinline__ Tok load_tok(const Args& a, const int* sSlots, int i, int b, int h) {
  Tok k;
  k.slot = sSlots[i];
  k.valid = k.slot > 0 && k.slot < a.P;   // PAD_SLOT_ID (-1) and the null block (0) are padding
  const int s = k.valid ? k.slot : 0;
  const int* cntp = a.cnt + (size_t)s * a.cnt_sst + h * 2;
  k.hc = cntp[0]; k.lim = cntp[1];
  k.dt = a.dt[b * a.H + h]; k.A = a.A[b * a.H + h]; k.trap = a.trap[b * a.H + h];
  return k;
}
__device__ __forceinline__ void clamp_tok(Tok& k, int L) {   // defensive: stay inside the ring
  k.hc = min(max(k.hc, 0), L - 1); k.lim = min(max(k.lim, 1), L);
}

__device__ __forceinline__ void issue_stage(const Args& a, char* stg, const StageLayout& sl, const Tok& k, int b, int h, int t) {
  if (!k.valid) return;
  const int H = a.H, slot = k.slot, hc = k.hc, n = hc + 2, cur = n - 1;
  __nv_bfloat16* sBk = reinterpret_cast<__nv_bfloat16*>(stg + sl.bk);
  __nv_bfloat16* sCb = reinterpret_cast<__nv_bfloat16*>(stg + sl.cb);
  __nv_bfloat16* sXb = reinterpret_cast<__nv_bfloat16*>(stg + sl.xb);
  // B' rows: token 0 (k_pool), the ring (this head's rotated rows, contiguous), then the current token's raw B and C
  for (int i = t; i < R * (S / 8); i += THREADS) {
    const int r = i >> 4, c = (i & 15) * 8;
    cp16(sBk + r * LDB + c, a.kp + (size_t)slot * a.kp_sst + ((size_t)r * H + h) * S + c);
    cp16(sBk + (cur * R + r) * LDB + c, a.Bm + (size_t)b * a.bc_bstride + r * S + c);
    cp16(sCb + r * LDB + c, a.Cm + (size_t)b * a.bc_bstride + r * S + c);
  }
  const __nv_bfloat16* br = a.br + (size_t)slot * a.br_sst + (size_t)h * a.L * R * S;
  for (int i = t; i < hc * R * (S / 8); i += THREADS) cp16(sBk + (R + (i >> 4)) * LDB + (i & 15) * 8, br + i * 8);
  const __nv_bfloat16* xr = a.xr + (size_t)slot * a.xr_sst + (size_t)h * a.L * D;
  if (t < D / 8) { cp16(sXb + t * 8, a.vp + (size_t)slot * a.vp_sst + h * D + t * 8); cp16(sXb + cur * D + t * 8, a.x + ((size_t)b * H + h) * D + t * 8); }
  for (int i = t; i < hc * (D / 8); i += THREADS) cp16(sXb + D + i * 8, xr + i * 8);
  if (t >= 64 && t < 64 + hc) cp16(stg + sl.scr + (t - 64) * 16, a.scr + (size_t)slot * a.scr_sst + (size_t)h * a.L * 4 + (t - 64) * 4);
  const int q = t - 96;   // the last warp: z row (8 chunks), carried phase (8), phase projection (4)
  if (q >= 0 && q < 8) cp16(stg + sl.z + q * 16, a.z + ((size_t)b * H + h) * D + q * 8);
  else if (q >= 8 && q < 16) cp16(stg + sl.angst + (q - 8) * 16, a.angle + (size_t)slot * a.ang_sst + (size_t)h * NA + (q - 8) * 4);
  else if (q >= 16 && q < 20) cp16(stg + sl.angp + (q - 16) * 16, a.angp + (size_t)b * a.angp_stride + (q - 16) * 8);
}

#ifndef REPLAY_MINB
#define REPLAY_MINB 4   // 4 CTAs/SM: 128 registers, no spills (5 spills; 3 is ~8% slower on B300)
#endif
template <bool ANCHOR_F32, bool READ_FP8>
__global__ void __launch_bounds__(THREADS, REPLAY_MINB) replay_step_kernel(const Args a) {
  extern __shared__ __align__(128) char smem[];
  const Layout lo(a.L, a.tpc);
  const StageLayout& sl = lo.st;
  float* sBias = reinterpret_cast<float*>(smem + lo.bias);
  float* sXp = reinterpret_cast<float*>(smem + lo.xproj);
  float* sZp = reinterpret_cast<float*>(smem + lo.zproj);
  float* sOp = reinterpret_cast<float*>(smem + lo.outproj);
  __half* sCh = reinterpret_cast<__half*>(smem + lo.ch);
  float* sG = reinterpret_cast<float*>(smem + lo.G);
  float* sW = reinterpret_cast<float*>(smem + lo.w);
  int* sSlots = reinterpret_cast<int*>(smem + lo.slots);
  __shared__ float sTh[NA], sCosC[NA], sSinC[NA];
  __shared__ float sDecay, sLcur;

  const int H = a.H, h = blockIdx.x, j0 = blockIdx.y, nb = gridDim.y, t = threadIdx.x;
  const int warp = t >> 5, lane = t & 31, g = lane >> 2, qd = lane & 3;
  const int p0 = warp * 16 + g, p1 = p0 + 8;
  const int mi = lane >> 3, li = lane & 7;
  const int ntok = (a.N - j0 + nb - 1) / nb;
  if (ntok <= 0) return;

  // ---- once per CTA: the head's constants, this CTA's slots, the zero rows
  for (int i = t; i < 2 * R * S / 4; i += THREADS) {
    const int qq = i / (R * S / 4), rem = i % (R * S / 4), r = rem / (S / 4), c = (rem % (S / 4)) * 4;
    cp16(sBias + (qq * R + r) * S + c, (qq ? a.bias_q : a.bias_k) + ((size_t)r * H + h) * S + c);
  }
  for (int i = t; i < 3 * R * D / 4; i += THREADS) {
    const int m = i / (R * D / 4), rem = i % (R * D / 4), r = rem / (D / 4), c = (rem % (D / 4)) * 4;
    const float* src = m == 0 ? a.xproj : m == 1 ? a.zproj : a.outproj;
    float* dst = m == 0 ? sXp : m == 1 ? sZp : sOp;
    cp16(dst + r * D + c, src + ((size_t)r * H + h) * D + c);
  }
  cp_commit();
  for (int i = t; i < ntok; i += THREADS) sSlots[i] = a.slots[j0 + i * nb];
  if (t < 4 * (S / 8)) {
    const int r = 4 + t / (S / 8), c = t % (S / 8);
#pragma unroll
    for (int s = 0; s < 2; ++s)
      *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(smem + lo.stage + s * sl.total + sl.cb) + r * LDB + c * 8) = make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(sCh + r * LDB + c * 8) = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  const float Dh = a.Dp[h];

  Tok cur_k = load_tok(a, sSlots, 0, j0, h);
  clamp_tok(cur_k, a.L);
  Tok nxt_k{};
  if (ntok > 1) nxt_k = load_tok(a, sSlots, 1, j0 + nb, h);
  issue_stage(a, smem + lo.stage, sl, cur_k, j0, h, t);
  cp_commit();
  for (int it = 0; it < ntok; ++it) {
    const int b = j0 + it * nb;
    char* stg = smem + lo.stage + (it & 1) * sl.total;
    // ---- prefetch: next token's buffers and anchor fragment, the scalars of the one after
    if (it + 1 < ntok) {
      clamp_tok(nxt_k, a.L);
      issue_stage(a, smem + lo.stage + ((it + 1) & 1) * sl.total, sl, nxt_k, b + nb, h, t);
    }
    cp_commit();
    Tok nn{};
    if (it + 2 < ntok) nn = load_tok(a, sSlots, it + 2, b + 2 * nb, h);

    const Tok k = cur_k;
    // fp8 checkpoint fragments straight into registers (no smem): lane (g, qd) takes columns [32 qd, 32 qd + 32) of
    // rows p0, p1; the mma k order is permuted consistently in C' (a dot product does not care), see the read below
    uint4 q8f[READ_FP8 ? 2 : 1][2];
    if constexpr (READ_FP8) {
      if (k.valid && k.hc + 1 < k.lim) {
        const uint8_t* qsrc = reinterpret_cast<const uint8_t*>(a.rd) + (size_t)k.slot * a.rd_sst + (size_t)h * D * S + qd * 32;
#pragma unroll
        for (int e = 0; e < 2; ++e)
#pragma unroll
          for (int hb = 0; hb < 2; ++hb) q8f[e][hb] = __ldcs(reinterpret_cast<const uint4*>(qsrc + (e ? p1 : p0) * S + hb * 16));
      }
    }
    __nv_bfloat16* yrow = a.y + ((size_t)b * H + h) * D;
    cp_wait<1>();
    __syncthreads();

    if (!k.valid) {
      if (t < D) yrow[t] = __float2bfloat16(0.f);
    } else {
      __nv_bfloat16* sBk = reinterpret_cast<__nv_bfloat16*>(stg + sl.bk);
      __nv_bfloat16* sCb = reinterpret_cast<__nv_bfloat16*>(stg + sl.cb);
      __nv_bfloat16* sXb = reinterpret_cast<__nv_bfloat16*>(stg + sl.xb);
      const float* sScr = reinterpret_cast<const float*>(stg + sl.scr);
      const __nv_bfloat16* sZ = reinterpret_cast<const __nv_bfloat16*>(stg + sl.z);
      const float* sAngSt = reinterpret_cast<const float*>(stg + sl.angst);
      const __nv_bfloat16* sAngP = reinterpret_cast<const __nv_bfloat16*>(stg + sl.angp);
      const int slot = k.slot, hc = k.hc;
      const bool flush = hc + 1 >= k.lim;
      const int n = hc + 2, cur = n - 1, nrows = n * R, nks = (nrows + 15) / 16;
      const float dt = k.dt, Av = k.A, tr = k.trap;
      const float alpha = __expf(Av * dt), lam = Av * dt, gam = tr * dt, bet = (1.f - tr) * dt * alpha;

      // ---- current phase (accurate sincos, wrapped to [0, 2 pi))
      if (t < NA) {
        float th = sAngSt[t] + tanhf(bf2f(sAngP[t])) * dt * PI;
        th -= TWO_PI * floorf(th / TWO_PI);
        sTh[t] = th; sincosf(th, &sSinC[t], &sCosC[t]);
      }
      for (int i = t; i < (nks * 16 - nrows) * (S / 8); i += THREADS) {   // pad rows of the last mma tile: zero
        const int r = nrows + i / (S / 8), c = i % (S / 8);
        *reinterpret_cast<uint4*>(sBk + r * LDB + c * 8) = make_uint4(0, 0, 0, 0);
      }
      // ---- window weights: thread k sums its own prefix of the log-decays (v1's order); token 0 = the checkpoint
      if (t < n) {
        float Lt = 0.f, Lt1 = 0.f, Lc = 0.f;
        for (int kk = 1; kk < cur; ++kk) {
          Lc += sScr[(kk - 1) * 4];
          if (kk == t) Lt = Lc;
          if (kk == t + 1) Lt1 = Lc;
        }
        Lc += lam;
        if (t == cur) Lt = Lc;
        if (t + 1 == cur) Lt1 = Lc;
        const float gk = t == 0 ? 0.f : t == cur ? gam : sScr[(t - 1) * 4 + 1];
        float wk = gk * __expf(Lc - Lt);
        if (t + 1 < n) wk += (t + 1 == cur ? bet : sScr[t * 4 + 2]) * __expf(Lc - Lt1);
        sW[t] = wk;
        if (t == 0) { sDecay = __expf(Lc); sLcur = Lc; }
      }
      __syncthreads();
      // ---- B'_cur = rot(B + bias_k), C' = rot(C + bias_q), in place, bf16-rounded (the ring rows are rotated already)
      constexpr int ntask = R * (HALF / 8);
      for (int i = t; i < 2 * ntask; i += THREADS) {
        const bool isC = i >= ntask;
        const int ii = isC ? i - ntask : i;
        const int kk = cur, r = (ii >> 3) & 3, j0c = (ii & 7) * 8;
        __nv_bfloat16* row = isC ? sCb + r * LDB : sBk + (kk * R + r) * LDB;
        const float* bias = sBias + ((isC ? R : 0) + r) * S;
        const uint4 ul = *reinterpret_cast<const uint4*>(row + j0c), uh = *reinterpret_cast<const uint4*>(row + j0c + HALF);
        const __nv_bfloat16* el = reinterpret_cast<const __nv_bfloat16*>(&ul);
        const __nv_bfloat16* eh = reinterpret_cast<const __nv_bfloat16*>(&uh);
        float vl[8], vh[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) { vl[q] = bf2f(el[q]) + bias[j0c + q]; vh[q] = bf2f(eh[q]) + bias[j0c + HALF + q]; }
        if (j0c < NA) {
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const float c = sCosC[j0c + q], sn = sSinC[j0c + q];
            const float l2 = vl[q] * c - vh[q] * sn, h2 = vl[q] * sn + vh[q] * c; vl[q] = l2; vh[q] = h2;
          }
        }
        uint4 pl, ph;
        pl.x = pack_bf16(vl[0], vl[1]); pl.y = pack_bf16(vl[2], vl[3]); pl.z = pack_bf16(vl[4], vl[5]); pl.w = pack_bf16(vl[6], vl[7]);
        ph.x = pack_bf16(vh[0], vh[1]); ph.y = pack_bf16(vh[2], vh[3]); ph.z = pack_bf16(vh[4], vh[5]); ph.w = pack_bf16(vh[6], vh[7]);
        *reinterpret_cast<uint4*>(row + j0c) = pl;
        *reinterpret_cast<uint4*>(row + j0c + HALF) = ph;
        if (isC) {   // f16 copy of the bf16-rounded C' (exact)
          pl.x = bf16x2_to_f16x2(pl.x); pl.y = bf16x2_to_f16x2(pl.y); pl.z = bf16x2_to_f16x2(pl.z); pl.w = bf16x2_to_f16x2(pl.w);
          ph.x = bf16x2_to_f16x2(ph.x); ph.y = bf16x2_to_f16x2(ph.y); ph.z = bf16x2_to_f16x2(ph.z); ph.w = bf16x2_to_f16x2(ph.w);
          *reinterpret_cast<uint4*>(sCh + r * LDB + j0c) = pl;
          *reinterpret_cast<uint4*>(sCh + r * LDB + j0c + HALF) = ph;
        }
      }
      __syncthreads();

      float xp[2][R];
#pragma unroll
      for (int r = 0; r < R; ++r) { xp[0][r] = sXp[r * D + p0]; xp[1][r] = sXp[r * D + p1]; }
      float o[2];
      const float decay = sDecay;
      if (!flush) {
        // ---- G = B' C'^T over 16-row tiles of the (k, r) rows
        for (int mt = warp; mt < nks; mt += 4) {
          float gc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int kk = 0; kk < S; kk += 16) {
            uint32_t af[4], bfr[2];
            ldsm_x4(af, sBk + (mt * 16 + (mi & 1) * 8 + li) * LDB + kk + (mi >> 1) * 8);
            ldsm_x2(bfr, sCb + (lane & 7) * LDB + kk + ((lane >> 3) & 1) * 8);
            mma_bf16(gc, af, bfr);
          }
          if (qd < 2) {
            const int m0 = mt * 16 + g, m1 = m0 + 8;
            if (m0 < nrows) { sG[m0 * R + 2 * qd] = gc[0]; sG[m0 * R + 2 * qd + 1] = gc[1]; }
            if (m1 < nrows) { sG[m1 * R + 2 * qd] = gc[2]; sG[m1 * R + 2 * qd + 1] = gc[3]; }
          }
        }
        // ---- checkpoint read: oc[p][r2] = sum_s S_c[p][s] C'[r2][s]
        float oc[2][R];
        if constexpr (READ_FP8) {
          // k step st covers columns 32 qd + 4 st + {0..3} on lane qd: A k-slots (2qd, 2qd+1) <- +{0, 1}, (2qd+8, 2qd+9)
          // <- +{2, 3}; the B fragment of lane (g, qd) takes the same columns of C' row g
          float c4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int st = 0; st < 8; ++st) {
            const uint32_t w0 = reinterpret_cast<const uint32_t*>(&q8f[0][st >> 2])[st & 3];
            const uint32_t w1 = reinterpret_cast<const uint32_t*>(&q8f[1][st >> 2])[st & 3];
            uint32_t af[4], bfr[2];
            af[0] = fp8x2_to_f16x2((uint16_t)(w0 & 0xffffu)); af[1] = fp8x2_to_f16x2((uint16_t)(w1 & 0xffffu));
            af[2] = fp8x2_to_f16x2((uint16_t)(w0 >> 16));    af[3] = fp8x2_to_f16x2((uint16_t)(w1 >> 16));
            const uint2 cb2 = *reinterpret_cast<const uint2*>(sCh + g * LDB + qd * 32 + st * 4);
            bfr[0] = cb2.x; bfr[1] = cb2.y;
            mma_f16(c4, af, bfr);
          }
          const float s0 = a.rd_scale[(size_t)slot * a.rds_sst + h * D + p0], s1 = a.rd_scale[(size_t)slot * a.rds_sst + h * D + p1];
#pragma unroll
          for (int r2 = 0; r2 < R; ++r2) {
            const int src = (g << 2) | (r2 >> 1);
            const float v0 = __shfl_sync(0xffffffffu, (r2 & 1) ? c4[1] : c4[0], src);
            const float v1 = __shfl_sync(0xffffffffu, (r2 & 1) ? c4[3] : c4[2], src);
            oc[0][r2] = v0 * s0; oc[1][r2] = v1 * s1;
          }
        } else {
#pragma unroll
          for (int e = 0; e < 2; ++e)
#pragma unroll
            for (int r2 = 0; r2 < R; ++r2) oc[e][r2] = 0.f;
#pragma unroll 4
          for (int jt = 0; jt < 16; ++jt) {
            const int nn2 = jt * 8 + 2 * qd;
            float2 cr[R];
#pragma unroll
            for (int r2 = 0; r2 < R; ++r2) cr[r2] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(sCb + r2 * LDB + nn2));
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const size_t off = (size_t)slot * a.anc_sst + ((size_t)h * D + (e ? p1 : p0)) * S + nn2;
              float2 v;
              if constexpr (ANCHOR_F32) v = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(a.anchor) + off);
              else v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const __nv_bfloat16*>(a.anchor) + off));
#pragma unroll
              for (int r2 = 0; r2 < R; ++r2) oc[e][r2] = fmaf(v.x, cr[r2].x, fmaf(v.y, cr[r2].y, oc[e][r2]));
            }
          }
#pragma unroll
          for (int e = 0; e < 2; ++e)
#pragma unroll
            for (int r2 = 0; r2 < R; ++r2) { oc[e][r2] += __shfl_xor_sync(0xffffffffu, oc[e][r2], 1); oc[e][r2] += __shfl_xor_sync(0xffffffffu, oc[e][r2], 2); }
        }
        __syncthreads();   // sG
        // ---- window term on tensor cores: W[p][(r, r2)] = sum_k (w_k x_k[p]) G_k[r][r2] (16 columns = two n8 tiles,
        // tile T holding r = 2T, 2T + 1), both operands split hi + lo in bf16 (~fp32 products); then
        // o[p][r2] = e^L oc[p][r2] + sum_r xproj[r][p] W[p][(r, r2)]
        float wc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int ks = 0; ks * 16 < n; ++ks) {
          uint32_t ahi[4], alo[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {   // a0: (p0, k, k+1), a1: (p1, ...), a2: (p0, k+8, k+9), a3: (p1, ...)
            const int k0 = ks * 16 + 2 * qd + (q >> 1) * 8, p = (q & 1) ? p1 : p0;
            const float u0 = k0 < n ? sW[k0] * bf2f(sXb[k0 * D + p]) : 0.f;
            const float u1 = k0 + 1 < n ? sW[k0 + 1] * bf2f(sXb[(k0 + 1) * D + p]) : 0.f;
            const float h0 = rbf(u0), h1 = rbf(u1);
            ahi[q] = pack_bf16(h0, h1); alo[q] = pack_bf16(u0 - h0, u1 - h1);
          }
#pragma unroll
          for (int T = 0; T < 2; ++T) {   // b0: (k = 2qd, 2qd + 1; col g), b1: (k = 2qd + 8, + 9; col g); col -> (r, r2)
            const int r = 2 * T + (g >> 2), r2 = g & 3;
            uint32_t bhi[2], blo[2];
#pragma unroll
            for (int hb = 0; hb < 2; ++hb) {
              const int k0 = ks * 16 + 2 * qd + hb * 8;
              const float g0 = k0 < n ? sG[(k0 * R + r) * R + r2] : 0.f;
              const float g1 = k0 + 1 < n ? sG[((k0 + 1) * R + r) * R + r2] : 0.f;
              const float h0 = rbf(g0), h1 = rbf(g1);
              bhi[hb] = pack_bf16(h0, h1); blo[hb] = pack_bf16(g0 - h0, g1 - h1);
            }
            mma_bf16(wc[T], alo, bhi); mma_bf16(wc[T], ahi, blo); mma_bf16(wc[T], ahi, bhi);
          }
        }
        // lane qd holds columns 2qd, 2qd + 1 of each tile: r = 2T + (qd >> 1), r2 = 2(qd & 1) + {0, 1}
        float v[2][2];
#pragma unroll
        for (int e = 0; e < 2; ++e)
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            float acc = xp[e][qd >> 1] * wc[0][2 * e + j] + xp[e][2 + (qd >> 1)] * wc[1][2 * e + j];
            v[e][j] = acc + __shfl_xor_sync(0xffffffffu, acc, 2);   // + the other r parity
          }
        const int src = (g << 2) | (qd >> 1);   // a lane whose (qd & 1) == this lane's qd >> 1 holds r2 = qd
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float s0 = __shfl_sync(0xffffffffu, v[e][0], src), s1 = __shfl_sync(0xffffffffu, v[e][1], src);
          o[e] = decay * (qd == 0 ? oc[e][0] : qd == 1 ? oc[e][1] : qd == 2 ? oc[e][2] : oc[e][3]) + ((qd & 1) ? s1 : s0);
        }
      } else {
        // ---- flush: S = e^L S_anchor + U B' with U[p][(k, r)] = w_k x_k[p] xproj[r][p] (mma, U split hi + lo)
        float acc[16][4];
#pragma unroll
        for (int jt = 0; jt < 16; ++jt) {
          const int nn2 = jt * 8 + 2 * qd;
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float2 v;
            if constexpr (ANCHOR_F32) {
              const size_t off = (size_t)slot * a.anc_sst + ((size_t)h * D + (e ? p1 : p0)) * S + nn2;
              v = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(a.anchor) + off);
            } else {
              const size_t off = (size_t)slot * a.anc_sst + ((size_t)h * D + (e ? p1 : p0)) * S + nn2;
              v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const __nv_bfloat16*>(a.anchor) + off));
            }
            acc[jt][2 * e] = decay * v.x; acc[jt][2 * e + 1] = decay * v.y;
          }
        }
        for (int ks = 0; ks < nks; ++ks) {
          uint32_t ahi[4], alo[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int e = q & 1, kk = ks * 16 + 2 * qd + (q >> 1) * 8, p = e ? p1 : p0;
            float u[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              const int m = kk + j, km = m >> 2;
              const int rr = (2 * qd + j) & 3;
              const float xpr = rr == 0 ? xp[e][0] : rr == 1 ? xp[e][1] : rr == 2 ? xp[e][2] : xp[e][3];
              u[j] = m < nrows ? sW[km] * bf2f(sXb[km * D + p]) * xpr : 0.f;
            }
            const float h0 = rbf(u[0]), h1 = rbf(u[1]);
            ahi[q] = pack_bf16(h0, h1); alo[q] = pack_bf16(u[0] - h0, u[1] - h1);
          }
#pragma unroll
          for (int jp = 0; jp < 8; ++jp) {
            uint32_t bfr[4];
            ldsm_x4_t(bfr, sBk + (ks * 16 + (mi & 1) * 8 + li) * LDB + jp * 16 + (mi >> 1) * 8);
            mma_bf16(acc[2 * jp], ahi, bfr); mma_bf16(acc[2 * jp], alo, bfr);
            mma_bf16(acc[2 * jp + 1], ahi, bfr + 2); mma_bf16(acc[2 * jp + 1], alo, bfr + 2);
          }
        }
        float oo[2][R];
#pragma unroll
        for (int e = 0; e < 2; ++e)
#pragma unroll
          for (int r2 = 0; r2 < R; ++r2) oo[e][r2] = 0.f;
#pragma unroll
        for (int jt = 0; jt < 16; ++jt) {
          const int nn2 = jt * 8 + 2 * qd;
#pragma unroll
          for (int r2 = 0; r2 < R; ++r2) {
            const float2 cr = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(sCb + r2 * LDB + nn2));
            oo[0][r2] = fmaf(acc[jt][0], cr.x, fmaf(acc[jt][1], cr.y, oo[0][r2]));
            oo[1][r2] = fmaf(acc[jt][2], cr.x, fmaf(acc[jt][3], cr.y, oo[1][r2]));
          }
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
#pragma unroll
          for (int r2 = 0; r2 < R; ++r2) { oo[e][r2] += __shfl_xor_sync(0xffffffffu, oo[e][r2], 1); oo[e][r2] += __shfl_xor_sync(0xffffffffu, oo[e][r2], 2); }
          o[e] = qd == 0 ? oo[e][0] : qd == 1 ? oo[e][1] : qd == 2 ? oo[e][2] : oo[e][3];
        }
        // ---- new checkpoint: anchor (fp32, or bf16 with stochastic rounding) and the fp8 read copy
        const uint32_t seed = hash32(__float_as_uint(sLcur) ^ (__float_as_uint(sTh[0]) * 0x9e3779b9u) ^ ((uint32_t)slot * 0x85ebca6bu));
#pragma unroll
        for (int jt = 0; jt < 16; ++jt) {
          const int nn2 = jt * 8 + 2 * qd;
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int p = e ? p1 : p0;
            const size_t off = (size_t)slot * a.anc_sst + ((size_t)h * D + p) * S + nn2;
            if constexpr (ANCHOR_F32) *reinterpret_cast<float2*>(reinterpret_cast<float*>(a.anchor) + off) = make_float2(acc[jt][2 * e], acc[jt][2 * e + 1]);
            else {
              const uint32_t rnd = hash32(((uint32_t)(h * D + p) * S + nn2) ^ seed);
              __nv_bfloat162 v; v.x = bf16_sr(acc[jt][2 * e], rnd); v.y = bf16_sr(acc[jt][2 * e + 1], rnd >> 16 | rnd << 16);
              *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(a.anchor) + off) = v;
            }
          }
        }
        if constexpr (READ_FP8) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float m = 0.f;
#pragma unroll
            for (int jt = 0; jt < 16; ++jt) m = fmaxf(m, fmaxf(fabsf(acc[jt][2 * e]), fabsf(acc[jt][2 * e + 1])));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1)); m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            const float sc = m > 0.f ? m / FP8_MAX : 1.f;
            const float inv = 1.f / sc;
            const int p = e ? p1 : p0;
            __nv_fp8_e4m3* dst = a.rd + (size_t)slot * a.rd_sst + ((size_t)h * D + p) * S;
#pragma unroll
            for (int jt = 0; jt < 16; ++jt) {
              const __nv_fp8x2_storage_t q = __nv_cvt_float2_to_fp8x2(make_float2(acc[jt][2 * e] * inv, acc[jt][2 * e + 1] * inv), __NV_SATFINITE, __NV_E4M3);
              *reinterpret_cast<__nv_fp8x2_storage_t*>(dst + jt * 8 + 2 * qd) = q;
            }
            if (qd == 0) a.rd_scale[(size_t)slot * a.rds_sst + h * D + p] = sc;
          }
        }
      }

      // ---- gated MIMO output: lane qd holds r = qd for rows p0, p1
      {
        float ys[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int p = e ? p1 : p0;
          const float xt = bf2f(sXb[cur * D + p]), zt = bf2f(sZ[p]);
          float orr = o[e] + Dh * xt * (qd == 0 ? xp[e][0] : qd == 1 ? xp[e][1] : qd == 2 ? xp[e][2] : xp[e][3]);
          const float zv = zt * sZp[qd * D + p];
          orr *= __fdividef(zv, 1.f + __expf(-zv));
          ys[e] = orr * sOp[qd * D + p];
          ys[e] += __shfl_xor_sync(0xffffffffu, ys[e], 1); ys[e] += __shfl_xor_sync(0xffffffffu, ys[e], 2);
        }
        if (qd == 0) { yrow[p0] = __float2bfloat16(ys[0]); yrow[p1] = __float2bfloat16(ys[1]); }
      }
      // ---- state bookkeeping
      int* cntp = a.cnt + (size_t)slot * a.cnt_sst + h * 2;
      if (t < NA) a.angle[(size_t)slot * a.ang_sst + h * NA + t] = sTh[t];
      if (flush) {
        if (t < R * S / 8) {   // B'_cur and x_cur become the token before the next window
          const int r = t / (S / 8), c = t % (S / 8);
          *reinterpret_cast<uint4*>(a.kp + (size_t)slot * a.kp_sst + ((size_t)r * H + h) * S + c * 8) =
              *reinterpret_cast<const uint4*>(sBk + (cur * R + r) * LDB + c * 8);
        }
        if (t < D / 8) *reinterpret_cast<uint4*>(a.vp + (size_t)slot * a.vp_sst + h * D + t * 8) = *reinterpret_cast<const uint4*>(sXb + cur * D + t * 8);
        if (t == 0) { cntp[0] = 0; cntp[1] = a.L; }
      } else {
        if (t < D / 8) *reinterpret_cast<uint4*>(a.xr + (size_t)slot * a.xr_sst + ((size_t)h * a.L + hc) * D + t * 8) = *reinterpret_cast<const uint4*>(sXb + cur * D + t * 8);
        if (t == 0) {
          *reinterpret_cast<float4*>(a.scr + (size_t)slot * a.scr_sst + ((size_t)h * a.L + hc) * 4) = make_float4(lam, gam, bet, 0.f);
          cntp[0] = hc + 1;
        }
        if (t >= 64) {   // this token's B' joins the ring
          const int i = t - 64, r = i >> 4, c = (i & 15) * 8;
          *reinterpret_cast<uint4*>(a.br + (size_t)slot * a.br_sst + (((size_t)h * a.L + hc) * R + r) * S + c) =
              *reinterpret_cast<const uint4*>(sBk + (cur * R + r) * LDB + c);
        }
      }
    }
    __syncthreads();   // this stage and the CTA scratch are reusable
    cur_k = nxt_k; nxt_k = nn;
  }
}
}  // namespace

void mamba3_replay_step(torch::Tensor angle, torch::Tensor anchor, c10::optional<torch::Tensor> rd, c10::optional<torch::Tensor> rd_scale,
                           torch::Tensor kp, torch::Tensor vp, torch::Tensor xr, torch::Tensor br, torch::Tensor scr, torch::Tensor cnt,
                           torch::Tensor A, torch::Tensor Bm, torch::Tensor Cm, torch::Tensor Dp, torch::Tensor x, torch::Tensor dt, torch::Tensor trap,
                           torch::Tensor xproj, torch::Tensor zproj, torch::Tensor outproj, torch::Tensor z, torch::Tensor bias_q, torch::Tensor bias_k,
                           torch::Tensor angp, torch::Tensor slots, torch::Tensor y, int64_t ctas_per_sm) {
  Args a{};
  const int H = x.size(1), L = xr.size(2), N = x.size(0);
  TORCH_CHECK(x.size(2) == D && anchor.size(2) == D && anchor.size(3) == S && kp.size(1) == R && angle.size(2) == NA, "Olala replay step: unsupported shapes");
  TORCH_CHECK(L >= 2 && L <= 32 && br.size(1) == H && br.size(2) == L && br.size(3) == R && br.size(4) == S && scr.size(2) == L && scr.size(3) == 4 && cnt.size(2) == 2);
  TORCH_CHECK(anchor.stride(3) == 1 && anchor.stride(2) == S && anchor.stride(1) == D * S, "anchor inner dims must be contiguous");
  TORCH_CHECK(kp.stride(1) == H * S && kp.stride(2) == S && vp.stride(1) == D && xr.stride(1) == L * D && br.stride(1) == L * R * S
              && br.stride(2) == R * S && br.stride(3) == S && scr.stride(1) == L * 4 && cnt.stride(1) == 2 && angle.stride(1) == NA);
  TORCH_CHECK(Bm.stride(1) == S && Bm.stride(2) == 1 && Cm.stride(0) == Bm.stride(0) && Cm.stride(1) == S && angp.stride(1) == 1);
  TORCH_CHECK((Bm.stride(0) * 2) % 16 == 0 && x.is_contiguous() && z.is_contiguous(), "B/C batch stride must be 16B aligned; x, z contiguous");
  TORCH_CHECK((angp.stride(0) * 2) % 16 == 0 && reinterpret_cast<uintptr_t>(angp.data_ptr()) % 16 == 0
              && (angle.stride(0) * 4) % 16 == 0, "phase rows must be 16B aligned");
  const bool f32 = anchor.dtype() == torch::kFloat32;
  TORCH_CHECK(f32 || anchor.dtype() == torch::kBFloat16);
  TORCH_CHECK((anchor.stride(0) * anchor.element_size()) % 16 == 0 && (kp.stride(0) * 2) % 16 == 0 && (vp.stride(0) * 2) % 16 == 0
              && (xr.stride(0) * 2) % 16 == 0 && (br.stride(0) * 2) % 16 == 0 && (scr.stride(0) * 4) % 16 == 0, "pool slot strides must be 16B aligned");
  if (rd) TORCH_CHECK(rd->stride(0) % 16 == 0 && rd->stride(1) == D * S && rd_scale.has_value());
  a.angle = angle.data_ptr<float>(); a.anchor = anchor.data_ptr();
  a.rd = rd ? reinterpret_cast<__nv_fp8_e4m3*>(rd->data_ptr()) : nullptr; a.rd_scale = rd_scale ? rd_scale->data_ptr<float>() : nullptr;
  a.kp = reinterpret_cast<__nv_bfloat16*>(kp.data_ptr()); a.vp = reinterpret_cast<__nv_bfloat16*>(vp.data_ptr());
  a.xr = reinterpret_cast<__nv_bfloat16*>(xr.data_ptr()); a.br = reinterpret_cast<__nv_bfloat16*>(br.data_ptr());
  a.scr = scr.data_ptr<float>(); a.cnt = cnt.data_ptr<int>();
  a.ang_sst = angle.stride(0); a.anc_sst = anchor.stride(0); a.rd_sst = rd ? rd->stride(0) : 0; a.rds_sst = rd_scale ? rd_scale->stride(0) : 0;
  a.kp_sst = kp.stride(0); a.vp_sst = vp.stride(0); a.xr_sst = xr.stride(0); a.br_sst = br.stride(0);
  a.scr_sst = scr.stride(0); a.cnt_sst = cnt.stride(0);
  a.P = anchor.size(0); a.H = H; a.L = L; a.N = N;
  a.A = A.data_ptr<float>(); a.Bm = reinterpret_cast<const __nv_bfloat16*>(Bm.data_ptr()); a.Cm = reinterpret_cast<const __nv_bfloat16*>(Cm.data_ptr());
  a.Dp = Dp.data_ptr<float>(); a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); a.dt = dt.data_ptr<float>(); a.trap = trap.data_ptr<float>();
  a.xproj = xproj.data_ptr<float>(); a.zproj = zproj.data_ptr<float>(); a.outproj = outproj.data_ptr<float>();
  a.z = reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()); a.bias_q = bias_q.data_ptr<float>(); a.bias_k = bias_k.data_ptr<float>();
  a.angp = reinterpret_cast<const __nv_bfloat16*>(angp.data_ptr()); a.slots = slots.data_ptr<int>(); a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  a.bc_bstride = Bm.stride(0); a.angp_stride = angp.stride(0);
  if (N == 0) return;
  // tokens of a head spread over nb CTAs: about ctas_per_sm resident CTAs per SM in total
  int dev = 0, nsm = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  const int nb = std::max(1, std::min<int>(N, (int)(nsm * std::max<int64_t>(1, ctas_per_sm) / H)));
  a.tpc = (N + nb - 1) / nb;
  const int smem = Layout(L, a.tpc).total;
  const dim3 grid(H, nb);
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, THREADS, smem, stream>>>(a);
  };
  const bool fp8 = rd.has_value();
  if (f32) { if (fp8) launch(replay_step_kernel<true, true>); else launch(replay_step_kernel<true, false>); }
  else { if (fp8) launch(replay_step_kernel<false, true>); else launch(replay_step_kernel<false, false>); }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("step", &mamba3_replay_step); }
