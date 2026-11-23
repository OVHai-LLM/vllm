// Olala Mamba-3 MIMO decode step for small batches (B <= 4), preamble fused, heads split over CTAs.
// Replaces _decode_preamble_bc_kernel + mamba3_step_f32_pipe_kernel after the in_proj GEMV (left to cuBLAS: it runs at
// the HBM limit). Reads the raw in_proj output zxdt_bc = [in_proj | in_proj_dyn](u) (bf16, [B, 6288 + 1056]).
//
// One CTA (4 warps) per (head h, PS = 8 state rows): 384 CTAs instead of the 48 of the step kernel, so the state
// (fp32, 4 KB per CTA and token) and the constants stream in on every SM; with PDL all of that is loaded before
// griddepcontrol.wait, under the GEMV. After the wait: dt / A / trap of the head (as _decode_preamble_kernel), warp r
// RMS-norms / biases / rotates row r of B and C (as _bc_norm_kernel + the step's preamble), warp b runs the update of
// token b exactly as mamba3_step_f32_pipe_kernel (fp32 state, tensor-core update with the 3-way bf16 split).
// Rows p of a head are independent; the head's new B' (k_pool) and angles are written by its last CTA once the
// others have read the old values (cnt[h], back to zero at the end of each launch).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
constexpr int H = 48, D = 64, S = 128, R = 4, NA = 32;
constexpr int NZX = 2 * D + 3;          // in_proj rows per head: z (D) | x (D) | dt | A | trap
constexpr int N1 = H * NZX;             // 6288
constexpr int NDYN = 2 * R * S + NA;    // 1056: B (R x S) | C (R x S) | angles
constexpr int PS = 8, SPLIT = D / PS, NOWN = 2 * PS + 3;
constexpr int NW = 4, THREADS = NW * 32;
constexpr int LDB = S + 8;

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(s)), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ int ld_acquire(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_release_add(int* p, int v) { asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
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
  if constexpr (NS == 3) split3_x2(x0, x1, *t0, *t1, *t2); else split2_x2(x0, x1, *t0, *t1);
}
// Q16 mantissas are stored offset binary (u = q + 32768): one PRMT builds the float 2^23 + u, minus the magic
__device__ __forceinline__ float i16lo(uint32_t w) { return __uint_as_float(__byte_perm(w, 0x4b00u, 0x5410)) - 8421376.f; }
__device__ __forceinline__ float i16hi(uint32_t w) { return __uint_as_float(__byte_perm(w, 0x4b00u, 0x5432)) - 8421376.f; }
__device__ __forceinline__ float softplus20(float v) { return v > 20.f ? v : logf(1.f + expf(v)); }

struct Args {
  const __nv_bfloat16* zb; long long zb_stride; int B;   // raw in_proj output [B, N1 + NDYN]
  int* cnt;                                              // [H], zero at rest
  void* ssm; float* ssc; __nv_bfloat16* kp; __nv_bfloat16* vp; float* angle; int P;
  long long ssm_sst, ssc_sst, kp_sst, vp_sst, ang_sst;
  const int* slots;
  const float* dt_bias; float a_floor; const float* wb; const float* wc; float eps;
  const float* Dp; const float* xproj; const float* zproj; const float* outproj; const float* bias_q; const float* bias_k;
  __nv_bfloat16* y; long long y_stride;
  int pdl;
};

template <int MAXB> struct __align__(16) Smem {
  __nv_bfloat16 sBk[MAXB][8][LDB];        // mma B operand of the update: rows 0..3 B'[r], 4..7 Bstate[r]
  __nv_bfloat16 sCb[MAXB][4][LDB];        // rotated C' (mma B operand of o = h C'^T)
  float sTheta[MAXB][NA];
};

// Q: Q16 state (int16 mantissas + one fp32 scale per half row, see mamba3_step_hc.cu; same arithmetic, bit-identical)
template <int MAXB, bool Q>
__global__ void __launch_bounds__(THREADS) mamba3_step_split_kernel(const Args a) {
  __shared__ Smem<MAXB> S_;
  constexpr int NS = Q ? 2 : 3;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, B = a.B;
  const int h = blockIdx.x / SPLIT, s = blockIdx.x % SPLIT;
  const int qd = lane & 3, p0 = lane >> 2, gp = s * PS + p0;   // step lanes: state row p0 of this CTA (global gp)
  // ---- independent of the in_proj GEMV (with PDL: under it): every load issued before anything waits
  int srow[MAXB]; bool valid[MAXB];
#pragma unroll
  for (int b = 0; b < MAXB; ++b) {
    const int slot = b < B ? a.slots[b] : -1;
    valid[b] = slot >= 0 && slot < a.P; srow[b] = valid[b] ? slot : 0;
  }
  float2 hst[16];                          // warp b: state row p0 of token b, columns 8 jt + 2 qd (Q16: raw mantissas)
  float hsc[2] = {1.f, 1.f};               // Q16: scales of the two half rows
  float xs = 0.f, xpj[R], zpj[R], opj[R];  // warp b: previous x (v_pool) and the MIMO projections of row gp
  if (warp < B) {
    if constexpr (Q) {
      const int16_t* st = reinterpret_cast<const int16_t*>(a.ssm) + (size_t)srow[warp] * a.ssm_sst + ((size_t)h * D + gp) * S + 2 * qd;
#pragma unroll
      for (int jt = 0; jt < 16; ++jt) { const uint32_t w = __ldcs(reinterpret_cast<const unsigned*>(st + jt * 8)); hst[jt] = make_float2(i16lo(w), i16hi(w)); }
      const float* sp = a.ssc + (size_t)srow[warp] * a.ssc_sst + ((size_t)h * D + gp) * 2;
      hsc[0] = sp[0]; hsc[1] = sp[1];
    } else {
      const float* st = reinterpret_cast<const float*>(a.ssm) + (size_t)srow[warp] * a.ssm_sst + ((size_t)h * D + gp) * S + 2 * qd;
#pragma unroll
      for (int jt = 0; jt < 16; ++jt) hst[jt] = __ldcs(reinterpret_cast<const float2*>(st + jt * 8));
    }
    xs = bf2f(a.vp[(size_t)srow[warp] * a.vp_sst + (size_t)h * D + gp]);
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const size_t off = ((size_t)r * H + h) * D + gp;
    xpj[r] = a.xproj[off]; zpj[r] = a.zproj[off]; opj[r] = a.outproj[off];
  }
  float bkv[4], bqv[4], wbv[4], wcv[4];    // warp r: bias_k / bias_q of row r, norm gains (1 + w), columns lane + 32 i
  __nv_bfloat16 kprev[MAXB][4];            // warp r: previous B'[r]
  float angst[MAXB];                       // lane l: carried angle l
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    bkv[i] = a.bias_k[((size_t)warp * H + h) * S + lane + 32 * i];
    bqv[i] = a.bias_q[((size_t)warp * H + h) * S + lane + 32 * i];
    wbv[i] = a.wb[lane + 32 * i] + 1.f; wcv[i] = a.wc[lane + 32 * i] + 1.f;
  }
#pragma unroll
  for (int b = 0; b < MAXB; ++b) {
    if (b < B) {
#pragma unroll
      for (int i = 0; i < 4; ++i) kprev[b][i] = a.kp[(size_t)srow[b] * a.kp_sst + ((size_t)warp * H + h) * S + lane + 32 * i];
      angst[b] = a.angle[(size_t)srow[b] * a.ang_sst + (size_t)h * NA + lane];
    }
  }
  const float dtb = a.dt_bias[h], Dh = a.Dp[h];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (a.pdl) asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  // ---- in_proj outputs: dt / A / trap of the head (every lane), B / C row `warp` and angle `lane` of every token,
  // z / x of row gp for the step lanes
  float raw[MAXB][3], fb[MAXB][4], fc[MAXB][4], angp[MAXB];
  float xp = 0.f, zp = 0.f;
#pragma unroll
  for (int b = 0; b < MAXB; ++b) {
    if (b < B) {
      const __nv_bfloat16* zr = a.zb + (size_t)b * a.zb_stride;
#pragma unroll
      for (int j = 0; j < 3; ++j) raw[b][j] = bf2f(zr[h * NZX + 2 * D + j]);
      const __nv_bfloat16* dy = zr + N1;
#pragma unroll
      for (int i = 0; i < 4; ++i) { fb[b][i] = bf2f(dy[warp * S + lane + 32 * i]); fc[b][i] = bf2f(dy[R * S + warp * S + lane + 32 * i]); }
      angp[b] = bf2f(dy[2 * R * S + lane]);
    }
  }
  if (warp < B) {
    const __nv_bfloat16* zr = a.zb + (size_t)warp * a.zb_stride + h * NZX;
    zp = bf2f(zr[gp]); xp = bf2f(zr[D + gp]);
  }
  // ---- preamble (as _decode_preamble_kernel / _bc_norm_kernel / the step's B, C prep), tokens interleaved
  float dt[MAXB], alpha[MAXB], gamma[MAXB], beta[MAXB];
#pragma unroll
  for (int b = 0; b < MAXB; ++b) {
    if (b < B) {
      dt[b] = softplus20(raw[b][0] + dtb);
      const float A = fminf(-softplus20(raw[b][1]), -a.a_floor);
      const float trap = 1.f / (1.f + expf(-raw[b][2]));
      alpha[b] = __expf(A * dt[b]); gamma[b] = trap * dt[b]; beta[b] = (1.f - trap) * dt[b] * alpha[b];
    }
  }
  float sb[MAXB], scc[MAXB];
#pragma unroll
  for (int b = 0; b < MAXB; ++b) {
    sb[b] = 0.f; scc[b] = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { if (b < B) { sb[b] += fb[b][i] * fb[b][i]; scc[b] += fc[b][i] * fc[b][i]; } }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1)
#pragma unroll
    for (int b = 0; b < MAXB; ++b) { sb[b] += __shfl_xor_sync(0xffffffffu, sb[b], o); scc[b] += __shfl_xor_sync(0xffffffffu, scc[b], o); }
#pragma unroll
  for (int b = 0; b < MAXB; ++b) {
    if (b < B) {
      const int r = warp;
      const float ib = 1.f / sqrtf(sb[b] / S + a.eps), ic = 1.f / sqrtf(scc[b] / S + a.eps);
      float th = angst[b] + tanhf(angp[b]) * dt[b] * 3.14159265358979323846f;
      th -= 6.283185307179586f * floorf(th / 6.283185307179586f);
      float sn, cs; sincosf(th, &sn, &cs);
      if (r == 0) S_.sTheta[b][lane] = th;
      float bv[4], cv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        bv[i] = bf2f(__float2bfloat16(fb[b][i] * ib * wbv[i])) + bkv[i];
        cv[i] = bf2f(__float2bfloat16(fc[b][i] * ic * wcv[i])) + bqv[i];
      }
      {   // columns (lane, lane + 64): the rotated pair (lane < NA); (lane + 32, lane + 96) are not rotated
        const float b0 = bv[0] * cs - bv[2] * sn, b1 = bv[0] * sn + bv[2] * cs, c0 = cv[0] * cs - cv[2] * sn, c1 = cv[0] * sn + cv[2] * cs;
        bv[0] = b0; bv[2] = b1; cv[0] = c0; cv[2] = c1;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        S_.sBk[b][r][lane + 32 * i] = __float2bfloat16(bv[i]);
        S_.sBk[b][4 + r][lane + 32 * i] = kprev[b][i];
        S_.sCb[b][r][lane + 32 * i] = __float2bfloat16(cv[i]);
      }
    }
  }
  __syncthreads();
  if (t == 0 && s != SPLIT - 1) red_release_add(a.cnt + h, 1);   // old B' / angles consumed

  if (warp < B) {   // warp b: the step of token b over this CTA's 8 state rows (rows 8..15 of the m16 tile are zero)
    const int b = warp;
    float al = 0.f, ga = 0.f, be = 0.f;
#pragma unroll
    for (int j = 0; j < MAXB; ++j) if (j == b) { al = alpha[j]; ga = gamma[j]; be = beta[j]; }
    float acc[16][4];
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      const float f = Q ? al * hsc[jt >> 3] : al;
      acc[jt][0] = f * hst[jt].x; acc[jt][1] = f * hst[jt].y; acc[jt][2] = 0.f; acc[jt][3] = 0.f;
    }
    uint32_t ah[3][4];
    {
      float av[2];
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const int kk = 2 * qd + jj;
        float xr = xpj[0];
#pragma unroll
        for (int r = 1; r < R; ++r) if ((kk & 3) == r) xr = xpj[r];
        av[jj] = (kk < 4) ? ga * xp * xr : be * xs * xr;
      }
      splitn_x2<NS>(av[0], av[1], &ah[0][0], &ah[1][0], &ah[2][0]);
#pragma unroll
      for (int q = 0; q < 3; ++q) { ah[q][1] = 0u; ah[q][2] = 0u; ah[q][3] = 0u; }
    }
    const int mi = lane >> 3, li = lane & 7;
#pragma unroll
    for (int jp = 0; jp < 8; ++jp) {
      uint32_t bfr[4];   // k rows 8..15 of the operand alias rows 0..7 (their A columns are zero)
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(bfr[0]), "=r"(bfr[1]), "=r"(bfr[2]), "=r"(bfr[3]) : "r"(smem_u32(&S_.sBk[b][li][jp * 16 + (mi >> 1) * 8])));
#pragma unroll
      for (int q = NS - 1; q >= 0; --q) { mma16816(acc[2 * jp], ah[q], bfr); mma16816(acc[2 * jp + 1], ah[q], bfr + 2); }
    }
    bool vb = false;
#pragma unroll
    for (int j = 0; j < MAXB; ++j) if (j == b) vb = valid[j];
    int slot_b = 0;
#pragma unroll
    for (int j = 0; j < MAXB; ++j) if (j == b) slot_b = srow[j];
    if constexpr (Q) {   // all lanes take part in the half-row max shuffles
      float m[2] = {0.f, 0.f};
#pragma unroll
      for (int jt = 0; jt < 16; ++jt) m[jt >> 3] = fmaxf(m[jt >> 3], fmaxf(fabsf(acc[jt][0]), fabsf(acc[jt][1])));
#pragma unroll
      for (int e = 0; e < 2; ++e) { m[e] = fmaxf(m[e], __shfl_xor_sync(0xffffffffu, m[e], 1)); m[e] = fmaxf(m[e], __shfl_xor_sync(0xffffffffu, m[e], 2)); }
      if (vb) {
        int16_t* dst = reinterpret_cast<int16_t*>(a.ssm) + (size_t)slot_b * a.ssm_sst + ((size_t)h * D + gp) * S + 2 * qd;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          // as mamba3_step_hc: approximate reciprocal (|h| * inv rounds to <= 32767), q = rint(h * inv) in one FFMA
          float sc = m[e] * 3.0518509475997192e-05f;
          if (!(sc >= 1.17549435e-38f)) sc = 1.f;   // zero (or a sub-FLT_MIN half: rcp.ftz would give inf) -> q = 0
          float inv; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(sc));
#pragma unroll
          for (int jt = 8 * e; jt < 8 * e + 8; ++jt)
            __stcs(reinterpret_cast<unsigned*>(dst + jt * 8), __byte_perm(__float_as_uint(__fmaf_rn(acc[jt][0], inv, 12615680.f)),
                                                                      __float_as_uint(__fmaf_rn(acc[jt][1], inv, 12615680.f)), 0x5410));
          if (qd == 0) a.ssc[(size_t)slot_b * a.ssc_sst + ((size_t)h * D + gp) * 2 + e] = sc;
        }
      }
    }
    if (vb) {
      if constexpr (!Q) {
        float* dst = reinterpret_cast<float*>(a.ssm) + (size_t)slot_b * a.ssm_sst + ((size_t)h * D + gp) * S + 2 * qd;
#pragma unroll
        for (int jt = 0; jt < 16; ++jt) __stcs(reinterpret_cast<float2*>(dst + jt * 8), make_float2(acc[jt][0], acc[jt][1]));
      }
      if (qd == 0) a.vp[(size_t)slot_b * a.vp_sst + (size_t)h * D + gp] = __float2bfloat16(xp);
    }
    float oc[4][4];
#pragma unroll
    for (int q = 0; q < 4; ++q) { oc[q][0] = 0.f; oc[q][1] = 0.f; oc[q][2] = 0.f; oc[q][3] = 0.f; }
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      uint32_t bfr[4];   // n rows 4..7 alias rows 0..3 (their outputs are unused)
      ldmatrix_x4(bfr, &S_.sCb[b][li & 3][c * 16 + (mi & 1) * 8]);
      const float* v[4] = {&acc[2 * c][0], &acc[2 * c][2], &acc[2 * c + 1][0], &acc[2 * c + 1][2]};
      uint32_t a3[3][4];
#pragma unroll
      for (int q = 0; q < 4; ++q) splitn_x2<NS>(v[q][0], v[q][1], &a3[0][q], &a3[1][q], &a3[2][q]);
#pragma unroll
      for (int q = NS - 1; q >= 0; --q) mma16816(oc[c & 3], a3[q], bfr);
    }
    float ys = 0.f;
    if (qd < 2) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int r = 2 * qd + j;
        float xr = xpj[0], zr = zpj[0], orr_ = opj[0];
#pragma unroll
        for (int rr = 1; rr < R; ++rr) if (r == rr) { xr = xpj[rr]; zr = zpj[rr]; orr_ = opj[rr]; }
        float orr = ((oc[0][j] + oc[1][j]) + (oc[2][j] + oc[3][j])) + Dh * xp * xr;
        const float zv = zp * zr;
        orr *= __fdividef(zv, 1.f + __expf(-zv));
        ys += orr * orr_;
      }
    }
    ys += __shfl_xor_sync(0xffffffffu, ys, 1);
    if (qd == 0) a.y[(size_t)b * a.y_stride + (size_t)h * D + gp] = __float2bfloat16(vb ? ys : 0.f);
  }
  // the head's last CTA writes the new B' and angles once the other CTAs have read the old ones
  if (s == SPLIT - 1) {
    if (t == 0) { while (ld_acquire(a.cnt + h) < SPLIT - 1) { } *(volatile int*)(a.cnt + h) = 0; }
    __syncthreads();
    for (int i = t; i < B * (R * S / 8 + NA); i += THREADS) {
      const int b = i / (R * S / 8 + NA), j = i % (R * S / 8 + NA);
      int slot = -1;
#pragma unroll
      for (int q = 0; q < MAXB; ++q) if (q == b && valid[q]) slot = srow[q];
      if (slot < 0) continue;
      if (j < R * S / 8) { const int r = j / (S / 8), c = j % (S / 8); *reinterpret_cast<uint4*>(a.kp + (size_t)slot * a.kp_sst + ((size_t)r * H + h) * S + c * 8) = *reinterpret_cast<const uint4*>(&S_.sBk[b][r][c * 8]); }
      else a.angle[(size_t)slot * a.ang_sst + (size_t)h * NA + (j - R * S / 8)] = S_.sTheta[b][j - R * S / 8];
    }
  }
}
}  // namespace

void mamba3_step_split(torch::Tensor zb, torch::Tensor cnt, torch::Tensor ssm_pool, std::optional<torch::Tensor> ssm_scale,
                       torch::Tensor k_pool, torch::Tensor v_pool, torch::Tensor angle_pool, torch::Tensor slots, torch::Tensor dt_bias, double a_floor, torch::Tensor wb,
                       torch::Tensor wc, double eps, torch::Tensor Dp, torch::Tensor xproj, torch::Tensor zproj, torch::Tensor outproj,
                       torch::Tensor bias_q, torch::Tensor bias_k, torch::Tensor y, bool pdl) {
  Args a;
  a.zb = reinterpret_cast<const __nv_bfloat16*>(zb.data_ptr()); a.zb_stride = zb.stride(0); a.B = zb.size(0);
  a.cnt = cnt.data_ptr<int>();
  const bool q16 = ssm_pool.dtype() == torch::kInt16;
  a.ssm = ssm_pool.data_ptr(); a.ssc = nullptr; a.ssc_sst = 0; a.kp = reinterpret_cast<__nv_bfloat16*>(k_pool.data_ptr()); a.vp = reinterpret_cast<__nv_bfloat16*>(v_pool.data_ptr());
  a.angle = angle_pool.data_ptr<float>(); a.P = ssm_pool.size(0);
  a.ssm_sst = ssm_pool.stride(0); a.kp_sst = k_pool.stride(0); a.vp_sst = v_pool.stride(0); a.ang_sst = angle_pool.stride(0);
  a.slots = slots.data_ptr<int>();
  a.dt_bias = dt_bias.data_ptr<float>(); a.a_floor = (float)a_floor; a.wb = wb.data_ptr<float>(); a.wc = wc.data_ptr<float>(); a.eps = (float)eps;
  a.Dp = Dp.data_ptr<float>(); a.xproj = xproj.data_ptr<float>(); a.zproj = zproj.data_ptr<float>(); a.outproj = outproj.data_ptr<float>();
  a.bias_q = bias_q.data_ptr<float>(); a.bias_k = bias_k.data_ptr<float>();
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr()); a.y_stride = y.stride(0); a.pdl = pdl ? 1 : 0;
  TORCH_CHECK(a.B >= 1 && a.B <= 4 && zb.size(1) == N1 + NDYN && zb.stride(1) == 1 && zb.dtype() == torch::kBFloat16 && cnt.numel() >= H);
  TORCH_CHECK((q16 || ssm_pool.dtype() == torch::kFloat32) && ssm_pool.stride(1) == D * S && ssm_pool.stride(2) == S && ssm_pool.stride(3) == 1);
  if (q16) {
    TORCH_CHECK(ssm_scale.has_value() && ssm_scale->dtype() == torch::kFloat32 && ssm_scale->size(0) == a.P && ssm_scale->stride(1) == D * 2 &&
                ssm_scale->stride(2) == 2 && ssm_scale->stride(3) == 1, "Q16 needs scales [P, H, D, 2] fp32");
    a.ssc = ssm_scale->data_ptr<float>(); a.ssc_sst = ssm_scale->stride(0);
  }
  TORCH_CHECK(k_pool.stride(1) == H * S && k_pool.stride(2) == S && v_pool.stride(1) == D && angle_pool.stride(1) == NA);
  TORCH_CHECK((a.kp_sst * 2) % 16 == 0 && (a.ssm_sst * 4) % 8 == 0);
  TORCH_CHECK(wb.dtype() == torch::kFloat32 && wc.dtype() == torch::kFloat32 && bias_q.dtype() == torch::kFloat32 && xproj.dtype() == torch::kFloat32);
  auto launch = [&](auto kernel) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(H * SPLIT); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = 0; cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = pdl ? 1 : 0;
    cudaLaunchKernelEx(&cfg, kernel, a);
  };
  if (q16) {
    if (a.B == 1) launch(mamba3_step_split_kernel<1, true>);
    else if (a.B == 2) launch(mamba3_step_split_kernel<2, true>);
    else launch(mamba3_step_split_kernel<4, true>);
  } else {
    if (a.B == 1) launch(mamba3_step_split_kernel<1, false>);
    else if (a.B == 2) launch(mamba3_step_split_kernel<2, false>);
    else launch(mamba3_step_split_kernel<4, false>);
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step", &mamba3_step_split, py::arg("zb"), py::arg("cnt"), py::arg("ssm"), py::arg("ssm_scale"), py::arg("kp"), py::arg("vp"), py::arg("angle"),
        py::arg("slots"), py::arg("dt_bias"), py::arg("a_floor"), py::arg("wb"), py::arg("wc"), py::arg("eps"), py::arg("Dp"),
        py::arg("xproj"), py::arg("zproj"), py::arg("outproj"), py::arg("bias_q"), py::arg("bias_k"), py::arg("y"), py::arg("pdl") = false);
}
