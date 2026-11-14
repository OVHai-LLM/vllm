// Olala Mamba-3 MIMO prefill output pass, Blackwell tcgen05 edition (sm_100 / sm_103).
// Same contract and math as the sm90 kernel in mamba3_scan_prefill.cu: the Triton phases of mimo_ssd.py give every
// 128-token chunk its entering state S_in (fp32, [P][N]) and the chunk-local rotary phases; this kernel walks the
// chunk's 32-token sub-blocks (128 MIMO rows, row = token * R + rank) and computes per sub-block:
//   O1  = qhat @ S                                         (inter-chunk, scaled by exp(da_cs) in the epilogue)
//   G   = qhat @ (khat * ts * e_j)^T, masked: earlier token -> exp(da_cs_i - da_cs_j) / e_j, same token -> gamma/ts/e_j
//   O2  = G @ PsiV ;  out[t] = sum_r (exp(da_cs) O1 + O2 + D PsiV)[t, r] * phi_r * silu(z_t * zeta_r)
//   S   = exp(da_cs_last) * S + (khat * ts * e_j)^T @ PsiV
// with e_j = exp(max(da_cs_rev_j, -80)) (da_cs / da_cs_rev: 32-token group cumsums from _ssd_prep).
// All four products are UMMA M=128 (cta_group::1) with fp32 accumulators in tensor memory (G 128 cols, S 64, and two
// O1 / O2 buffers of 64 + 64; S stays in TMEM across the chunk's sub-blocks). Persistent CTA, work item = chunk * H +
// head; warp roles:
//   * warps 9-19 (producers): turn the raw sub-block, landed by TMA (Q / K rows straight into the 128B-swizzled operand
//     tiles, V, z, phases), in place into the bf16 operands (bias, rotation, trap scale, PsiV), the 32 x 32 token mask,
//     the cos / sin table and the gate constants, into a two-deep ring;
//   * warp 8 (one thread): issues the UMMAs, and the raw Q / K / V / phase loads of the sub-block two ahead as soon as
//     the products of the stage's last user are complete;
//   * warps 0-3 (epilogue A, row = 32 w + lane): mask G into bf16 over the consumed qhat tile, decay S, bf16 copy of
//     the updated S (B operand of the next qhat @ S);
//   * warps 4-7 (epilogue B): outputs of the other O1 / O2 buffer (D skip, gate, rank sum by a transposing
//     butterfly), so they overlap the next sub-block's products.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int R = 4, N = 128, P = 64, NQ = 32, C = 32, ROWS = C * R;  // 128 rows per sub-block
constexpr int NEA = 4, NEB = 4, MMA_WARP = 8, PROD0 = 9, NPROD = 11;
constexpr int THREADS = (PROD0 + NPROD) * 32;  // 640
constexpr int PT = NPROD * 32;                 // 352 producer threads
constexpr int TILE = ROWS * 64;                // one 128-row x 64-col bf16 tile (128B rows, SW128)
constexpr int TILE_B = TILE * 2;               // 16 KB
constexpr float LOG_EMIN = 80.f;
constexpr int TMEM_COLS = 512;
constexpr int TM_G = 0, TM_S = 128, TM_O1 = 192, TM_O2 = 256;  // O1 / O2 of buffer b at + 128 b
constexpr int BAR_PROD = 1;
constexpr int PP = P + 4;
constexpr unsigned TX_BYTES = 4 * TILE_B + 2 * C * P * 2 + C * NQ * 4;

struct __align__(1024) Stage {
  __nv_bfloat16 q[2 * TILE];   // Q -> qhat [n half][row][64] (A, K-major); then the masked G [j half][row][64]
  __nv_bfloat16 k[2 * TILE];   // K -> khat * ts * e_j [n half][row][64] (B of G K-major; A of the S update MN-major)
  __nv_bfloat16 pv[TILE];      // PsiV [row][p] (B, MN-major)
  __nv_bfloat16 v[C * P];      // raw V [tok][p] (SW128)
  __nv_bfloat16 z[C * P];      // raw z [tok][p] (SW128)
  float th[C * NQ];            // chunk-local phases [tok][nq]; then the cos table
  float sn[C * NQ];            // sin table
  float coef[C * C];           // token mask [i][j]
  float phi[R * PP], zeta[R * PP];  // bf16-rounded phi, zeta; rows padded (conflict-free across ranks)
  float dcs[C], dcsr[C], ts[C], gam[C];
  float dsk;
  int nv;
};
struct Smem {
  Stage stage[2];
  __nv_bfloat16 sb[TILE];      // state, bf16 [n][p] (B of qhat @ S, MN-major)
  __nv_bfloat16 qb[R * N], kb[R * N], psi[R * P];
  float a0[NQ];
  unsigned long long tfull[2], full[2], qk_free[2], empty[2], o_done[2], o_free[2], g_full, sb_ready, gm_ready;
  uint32_t tmem_base;
};
struct Maps { CUtensorMap q, k, v, z, th; };

__device__ __forceinline__ int swz(int row, int col) {  // 128B swizzle, 64-element rows
  return row * 64 + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7);
}
__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(void* mbar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(count));
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
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
// smem matrix descriptor (sm_100): addr, LBO, SBO (>>4), version 1, 128B swizzle
__device__ __forceinline__ uint64_t udesc(const void* p, uint32_t lbo, uint32_t sbo) {
  uint64_t d = (uint64_t)((smem_u32(p) & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
__device__ __forceinline__ uint64_t kdesc(const void* p) { return udesc(p, 16, 1024); }           // K-major SW128
__device__ __forceinline__ uint64_t mdesc(const void* p) { return udesc(p, TILE_B, 1024); }       // MN-major SW128
// instruction descriptor: f32 accumulate, bf16 A/B
__host__ __device__ constexpr uint32_t idesc(int M, int Nn, int a_mn, int b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) |
         ((uint32_t)(Nn >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
__device__ __forceinline__ void umma(uint32_t tmem_d, uint64_t a, uint64_t b, uint32_t id, uint32_t accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               :: "r"(tmem_d), "l"(a), "l"(b), "r"(id), "r"(accum) : "memory");
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
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

union U8 { uint4 u; __nv_bfloat16 h[8]; __nv_bfloat162 h2[4]; };
__device__ __forceinline__ void mbar_arrive_expect_tx(void* mbar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, int c0, int c1, void* mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tma3d(void* dst, const CUtensorMap* m, int c0, int c1, int c2, void* mbar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void cp4(void* dst, const void* src, bool ok) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(smem_u32(dst)), "l"(src), "r"(ok ? 4 : 0));
}
__device__ __forceinline__ void cp_commit_wait() { asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory"); }


// Q / K rows, V and the phases of one sub-block (the producer's raw inputs) into stage s; z follows from the producer
// once the outputs are done with the stage.
__device__ __forceinline__ void issue_qk(Smem& S, Stage& sg, int s, const Maps& maps, int t0, int h) {
  mbar_arrive_expect_tx(&S.tfull[s], TX_BYTES);
  tma2d(&sg.q[0], &maps.q, 0, t0 * R, &S.tfull[s]);
  tma2d(&sg.q[TILE], &maps.q, 64, t0 * R, &S.tfull[s]);
  tma2d(&sg.k[0], &maps.k, 0, t0 * R, &S.tfull[s]);
  tma2d(&sg.k[TILE], &maps.k, 64, t0 * R, &S.tfull[s]);
  tma3d(sg.v, &maps.v, 0, h, t0, &S.tfull[s]);
  tma3d(sg.th, &maps.th, 0, h, t0, &S.tfull[s]);
}

// ---------------------------------------------------------------------------------------------- producer
// In place: Q / K tiles (TMA-loaded raw rows) -> qhat, khat * ts * e_j; raw V -> PsiV; the token mask, the cos / sin
// table of the absolute phases and the gate constants. In the Q / K pass a thread owns (row, cr): the rotated pair of
// 16-byte pieces (8 cr, 8 cr + 64) and the plain pair (32 + 8 cr, 96 + 8 cr), so no lane idles during the rotation.
__device__ __forceinline__ void produce(Smem& S, Stage& sg, int pt, int c, int h, int nv, bool last_sb, int last_seq,
                                        int H, const float* __restrict__ phi, const float* __restrict__ zeta,
                                        __nv_bfloat16* __restrict__ fin_k) {
  {
    const int lane = pt & 31;  // = tj along the stride (PT is a multiple of 32); lane t also holds token t's terms
    const float lej = fmaxf(sg.dcsr[lane], -LOG_EMIN), elj = __expf(-lej), dj = sg.dcs[lane];
    const float tsl = lane < nv ? rbf(sg.ts[lane]) : 0.f;
    const float dgl = (lane < nv && tsl > 0.f) ? __fdividef(sg.gam[lane], tsl) : 0.f;
    for (int idx = pt; idx < C * C; idx += PT) {
      const int ti = idx / C;
      const float di = __shfl_sync(0xffffffffu, dj, ti), dgi = __shfl_sync(0xffffffffu, dgl, ti);
      sg.coef[idx] = ti > lane ? __expf(fminf(di - dj, 0.f) - lej) : (ti == lane ? dgi * elj : 0.f);
    }
  }
  for (int idx = pt; idx < C * NQ; idx += PT) {
    float th = sg.th[idx] + S.a0[idx & (NQ - 1)];
    th = th - 6.283185307179586f * floorf(th * 0.15915494309189535f);
    float sv, cv;
    __sincosf(th, &sv, &cv);
    sg.th[idx] = cv;
    sg.sn[idx] = sv;
  }
  for (int idx = pt; idx < R * P; idx += PT) {
    sg.phi[(idx >> 6) * PP + (idx & 63)] = rbf(phi[h * R * P + idx]);
    sg.zeta[(idx >> 6) * PP + (idx & 63)] = 0.5f * zeta[h * R * P + idx];
  }
  for (int u = pt; u < ROWS * 8; u += PT) {  // PsiV, (row, 16-byte piece)
    const int row = u >> 3, ci = u & 7, tok = row >> 2, rr = row & 3;
    U8 va, ps;
    va.u = tok < nv ? *reinterpret_cast<const uint4*>(&sg.v[swz(tok, 8 * ci)]) : make_uint4(0, 0, 0, 0);
    ps.u = *reinterpret_cast<const uint4*>(&S.psi[rr * P + 8 * ci]);
#pragma unroll
    for (int j = 0; j < 4; ++j) va.h2[j] = __hmul2(va.h2[j], ps.h2[j]);
    *reinterpret_cast<uint4*>(&sg.pv[swz(row, 8 * ci)]) = va.u;
  }
  bar_sync(BAR_PROD, PT);  // cos / sin table complete
  // unit u -> (row, cr): a quarter-warp covers rows r and r + 4 (cr 0..3 each), so its eight 16-byte pieces fall in
  // eight distinct swizzle slots; rows advance by PT / 4 (a multiple of 8) per stride, so rr and cr stay fixed
  const int cr = pt & 3, rr = (pt >> 3) & 3;
  auto unit_row = [](int u) { return (u >> 5) * 8 + ((u >> 2) & 1) * 4 + ((u >> 3) & 3); };
  const int cA = 8 * cr, cB = 32 + 8 * cr;  // rotated piece pair (cA, cA + 64), plain pair (cB, cB + 64)
  U8 bq[4], bk[4];
  bq[0].u = *reinterpret_cast<const uint4*>(&S.qb[rr * N + cA]);
  bq[1].u = *reinterpret_cast<const uint4*>(&S.qb[rr * N + cA + 64]);
  bq[2].u = *reinterpret_cast<const uint4*>(&S.qb[rr * N + cB]);
  bq[3].u = *reinterpret_cast<const uint4*>(&S.qb[rr * N + cB + 64]);
  bk[0].u = *reinterpret_cast<const uint4*>(&S.kb[rr * N + cA]);
  bk[1].u = *reinterpret_cast<const uint4*>(&S.kb[rr * N + cA + 64]);
  bk[2].u = *reinterpret_cast<const uint4*>(&S.kb[rr * N + cB]);
  bk[3].u = *reinterpret_cast<const uint4*>(&S.kb[rr * N + cB + 64]);
#pragma unroll 1
  for (int u = pt; u < ROWS * 4; u += PT) {
    const int row = unit_row(u), tok = row >> 2;
    const bool ok = tok < nv;
    const float tse = ok ? rbf(sg.ts[tok]) * __expf(fmaxf(sg.dcsr[tok], -LOG_EMIN)) : 0.f;
    const __nv_bfloat162 tse2 = __float2bfloat162_rn(tse);
    const int oA = swz(row, cA), oB = swz(row, cB);
    U8 q[4], k[4];
    q[0].u = *reinterpret_cast<const uint4*>(&sg.q[oA]);
    q[1].u = *reinterpret_cast<const uint4*>(&sg.q[TILE + oA]);
    q[2].u = *reinterpret_cast<const uint4*>(&sg.q[oB]);
    q[3].u = *reinterpret_cast<const uint4*>(&sg.q[TILE + oB]);
    k[0].u = *reinterpret_cast<const uint4*>(&sg.k[oA]);
    k[1].u = *reinterpret_cast<const uint4*>(&sg.k[TILE + oA]);
    k[2].u = *reinterpret_cast<const uint4*>(&sg.k[oB]);
    k[3].u = *reinterpret_cast<const uint4*>(&sg.k[TILE + oB]);
    const float4* cp = reinterpret_cast<const float4*>(&sg.th[tok * NQ + cA]);
    const float4* sp = reinterpret_cast<const float4*>(&sg.sn[tok * NQ + cA]);
    const float4 ca = cp[0], cb = cp[1], sa = sp[0], sb = sp[1];
    const float cn[8] = {ca.x, ca.y, ca.z, ca.w, cb.x, cb.y, cb.z, cb.w};
    const float snv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        q[u].h2[j] = __hadd2(q[u].h2[j], bq[u].h2[j]);
        k[u].h2[j] = __hadd2(k[u].h2[j], bk[u].h2[j]);
      }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float c_a = cn[2 * j], c_b = cn[2 * j + 1], s_a = snv[2 * j], s_b = snv[2 * j + 1];
      const float2 x0 = __bfloat1622float2(q[0].h2[j]), x1 = __bfloat1622float2(q[1].h2[j]);
      const float2 y0 = __bfloat1622float2(k[0].h2[j]), y1 = __bfloat1622float2(k[1].h2[j]);
      q[0].h2[j] = __floats2bfloat162_rn(c_a * x0.x - s_a * x1.x, c_b * x0.y - s_b * x1.y);
      q[1].h2[j] = __floats2bfloat162_rn(s_a * x0.x + c_a * x1.x, s_b * x0.y + c_b * x1.y);
      k[0].h2[j] = __floats2bfloat162_rn(c_a * y0.x - s_a * y1.x, c_b * y0.y - s_b * y1.y);
      k[1].h2[j] = __floats2bfloat162_rn(s_a * y0.x + c_a * y1.x, s_b * y0.y + c_b * y1.y);
    }
    if (last_sb && last_seq >= 0 && tok == nv - 1) {
      __nv_bfloat16* fk = fin_k + ((long)(c * R + rr) * H + h) * N;
      *reinterpret_cast<uint4*>(fk + cA) = k[0].u;
      *reinterpret_cast<uint4*>(fk + cA + 64) = k[1].u;
      *reinterpret_cast<uint4*>(fk + cB) = k[2].u;
      *reinterpret_cast<uint4*>(fk + cB + 64) = k[3].u;
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (!ok) q[u].u = make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int j = 0; j < 4; ++j) k[u].h2[j] = __hmul2(k[u].h2[j], tse2);
    }
    *reinterpret_cast<uint4*>(&sg.q[oA]) = q[0].u;
    *reinterpret_cast<uint4*>(&sg.q[TILE + oA]) = q[1].u;
    *reinterpret_cast<uint4*>(&sg.q[oB]) = q[2].u;
    *reinterpret_cast<uint4*>(&sg.q[TILE + oB]) = q[3].u;
    *reinterpret_cast<uint4*>(&sg.k[oA]) = k[0].u;
    *reinterpret_cast<uint4*>(&sg.k[TILE + oA]) = k[1].u;
    *reinterpret_cast<uint4*>(&sg.k[oB]) = k[2].u;
    *reinterpret_cast<uint4*>(&sg.k[TILE + oB]) = k[3].u;
  }
}

__global__ void __launch_bounds__(THREADS, 1) mamba3_scan_prefill_tc_kernel(
    const __grid_constant__ Maps maps,
    const float* __restrict__ qbias, const float* __restrict__ kbias,          // (H, R, N)
    const float* __restrict__ psi, const float* __restrict__ phi, const float* __restrict__ zeta,  // (H, R, P)
    const float* __restrict__ Dskip,                                           // (H,)
    const float* __restrict__ ain,                                             // (NC, H, NQ)
    const float* __restrict__ dcs32, const float* __restrict__ dcsr32,         // (H, T)
    const float* __restrict__ ts_g, const float* __restrict__ gam_g,           // (H, T)
    const int* __restrict__ ch_start, const int* __restrict__ ch_len, const int* __restrict__ ch_last,
    const float* __restrict__ sin_t,                                           // (NC, H, P, N)
    __nv_bfloat16* __restrict__ out,                                           // (T, H, P)
    __nv_bfloat16* __restrict__ fin_k,                                         // (NC, R, H, N), chunks ending a sequence
    int T, int H, int NC) {
  extern __shared__ unsigned char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023));
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nitems = NC * H;
  if (blockIdx.x >= nitems) return;

  if (warp == MMA_WARP) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(smem_u32(&S.tmem_base)), "n"(TMEM_COLS));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
  }
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) {
      mbar_init(&S.tfull[i], 1); mbar_init(&S.full[i], NPROD); mbar_init(&S.qk_free[i], 1); mbar_init(&S.empty[i], NEB);
      mbar_init(&S.o_done[i], 1); mbar_init(&S.o_free[i], NEB);
    }
    mbar_init(&S.g_full, 1); mbar_init(&S.sb_ready, NEA); mbar_init(&S.gm_ready, NEA);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = S.tmem_base;

  if (warp >= PROD0) {
    // ================= producers =================
    const int pt = tid - PROD0 * 32;
    int gi = 0;
    for (int item = blockIdx.x; item < nitems; item += gridDim.x) {
      const int c = item / H, h = item % H;
      const int cs = ch_start[c], cl = ch_len[c], last_seq = ch_last[c], nsb = (cl + C - 1) / C;
      bar_sync(BAR_PROD, PT);  // every producer is done with the previous head's constants
      for (int j = pt; j < R * N; j += PT) {
        S.qb[j] = __float2bfloat16(qbias[h * R * N + j]);
        S.kb[j] = __float2bfloat16(kbias[h * R * N + j]);
      }
      for (int j = pt; j < R * P; j += PT) S.psi[j] = __float2bfloat16(psi[h * R * P + j]);
      if (pt < NQ) S.a0[pt] = ain[(c * H + h) * NQ + pt];
      for (int it = 0; it < nsb; ++it, ++gi) {
        const int s = gi & 1, t0 = cs + it * C, nv = min(C, cl - it * C);
        Stage& sg = S.stage[s];
        if (pt == 0) {
          if (gi < 2) issue_qk(S, sg, s, maps, t0, h);  // later sub-blocks: issued by the MMA thread
        }
        if (gi >= 2) mbar_wait(&S.empty[s], ((gi >> 1) + 1) & 1);  // the outputs are done with it too
        if (pt == 0) {
          tma3d(sg.z, &maps.z, 0, h, t0, &S.tfull[s]);
          sg.dsk = Dskip[h];
          sg.nv = nv;
        }
        if (pt < 4 * C) {
          const int a = pt >> 5, j = pt & 31;
          const bool ok = j < nv;
          const float* src = a == 0 ? dcs32 : a == 1 ? dcsr32 : a == 2 ? ts_g : gam_g;
          float* dst = a == 0 ? sg.dcs : a == 1 ? sg.dcsr : a == 2 ? sg.ts : sg.gam;
          cp4(dst + j, src + (ok ? (long)h * T + t0 + j : 0), ok);
        }
        cp_commit_wait();
        mbar_wait(&S.tfull[s], (gi >> 1) & 1);
        bar_sync(BAR_PROD, PT);  // scalars, constants and raw tiles visible to every producer
        produce(S, sg, pt, c, h, nv, it + 1 == nsb, last_seq, H, phi, zeta, fin_k);
        fence_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&S.full[s]);
      }
    }
  } else if (warp == MMA_WARP) {
    // ================= MMA issuer =================
    if (lane == 0) {
      constexpr uint32_t ID_G = idesc(128, 128, 0, 0), ID_O = idesc(128, 64, 0, 1), ID_S = idesc(128, 64, 1, 1);
      // cursor two sub-blocks ahead: its raw inputs go out as soon as the products of the stage's last user are done
      int la_item = blockIdx.x, la_it = 0, la_nsb = (ch_len[la_item / H] + C - 1) / C;
      auto advance = [&]() {
        if (++la_it == la_nsb) {
          la_it = 0;
          la_item += gridDim.x;
          if (la_item < nitems) la_nsb = (ch_len[la_item / H] + C - 1) / C;
        }
      };
      advance();
      advance();
      int gi = 0;
      for (int item = blockIdx.x; item < nitems; item += gridDim.x) {
        const int nsb = (ch_len[item / H] + C - 1) / C;
        for (int it = 0; it < nsb; ++it, ++gi) {
          const int s = gi & 1, b = gi & 1;
          Stage& sg = S.stage[s];
          mbar_wait(&S.full[s], (gi >> 1) & 1);
          mbar_wait(&S.sb_ready, gi & 1);
          if (gi >= 2) mbar_wait(&S.o_free[b], ((gi >> 1) + 1) & 1);
          tc_fence_after();
#pragma unroll
          for (int kk = 0; kk < 8; ++kk)  // G = qhat @ k^T
            umma(tmem + TM_G, kdesc(&sg.q[(kk >> 2) * TILE + (kk & 3) * 16]), kdesc(&sg.k[(kk >> 2) * TILE + (kk & 3) * 16]),
                 ID_G, kk > 0);
#pragma unroll
          for (int kk = 0; kk < 8; ++kk)  // O1 = qhat @ S
            umma(tmem + TM_O1 + 128 * b, kdesc(&sg.q[(kk >> 2) * TILE + (kk & 3) * 16]), mdesc(&S.sb[kk * 16 * 64]), ID_O, kk > 0);
          umma_commit(&S.g_full);
          mbar_wait(&S.gm_ready, gi & 1);
          tc_fence_after();
#pragma unroll
          for (int kk = 0; kk < 8; ++kk)  // O2 = G @ PsiV (G over the qhat tile)
            umma(tmem + TM_O2 + 128 * b, kdesc(&sg.q[(kk >> 2) * TILE + (kk & 3) * 16]), mdesc(&sg.pv[kk * 16 * 64]), ID_O, kk > 0);
#pragma unroll
          for (int kk = 0; kk < 8; ++kk)  // S += k^T @ PsiV
            umma(tmem + TM_S, mdesc(&sg.k[kk * 16 * 64]), mdesc(&sg.pv[kk * 16 * 64]), ID_S, 1);
          umma_commit(&S.o_done[b]);
          umma_commit(&S.qk_free[s]);
          if (la_item < nitems) {
            mbar_wait(&S.qk_free[s], (gi >> 1) & 1);
            issue_qk(S, sg, s, maps, ch_start[la_item / H] + la_it * C, la_item % H);
            advance();
          }
        }
      }
    }
    __syncwarp();
  } else if (warp < NEA) {
    {
    // ================= epilogue A (row = 32 w + lane): state and masked G =================
    const int row = 32 * warp + lane, tok = row >> 2;
    const uint32_t tl = tmem + ((uint32_t)(32 * warp) << 16);
    int gi = 0;
    for (int item = blockIdx.x; item < nitems; item += gridDim.x) {
      const int c = item / H, h = item % H;
      const int nsb = (ch_len[c] + C - 1) / C;
      {  // entering state -> TMEM (fp32) and its bf16 copy
        const float* sb = sin_t + (long)(c * H + h) * P * N + row;
#pragma unroll
        for (int ph = 0; ph < 2; ++ph) {
          float v[32];
#pragma unroll
          for (int p = 0; p < 32; ++p) v[p] = sb[(32 * ph + p) * N];
          tmem_st32(tl + TM_S + 32 * ph, v);
#pragma unroll
          for (int ch = 0; ch < 4; ++ch) {
            uint4 u = make_uint4(pack_bf16(v[8 * ch], v[8 * ch + 1]), pack_bf16(v[8 * ch + 2], v[8 * ch + 3]),
                                 pack_bf16(v[8 * ch + 4], v[8 * ch + 5]), pack_bf16(v[8 * ch + 6], v[8 * ch + 7]));
            *reinterpret_cast<uint4*>(&S.sb[swz(row, 32 * ph + 8 * ch)]) = u;
          }
        }
        tmem_wait_st();
        fence_async();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&S.sb_ready);
      }
      for (int it = 0; it < nsb; ++it, ++gi) {
        const int s = gi & 1, b = gi & 1;
        Stage& sg = S.stage[s];
        mbar_wait(&S.full[s], (gi >> 1) & 1);
        const int nv = sg.nv;
        mbar_wait(&S.g_full, gi & 1);
        tc_fence_after();
        {  // masked G -> bf16 over the qhat tile (both products reading it are complete)
          const float* cfr = &sg.coef[tok * C];
#pragma unroll
          for (int jc = 0; jc < 4; ++jc) {
            float g[32];
            tmem_ld32(tl + TM_G + 32 * jc, g);
            tmem_wait_ld();
#pragma unroll
            for (int ch = 0; ch < 4; ++ch) {
              const float ca = cfr[8 * jc + 2 * ch], cb = cfr[8 * jc + 2 * ch + 1];
              uint4 u = make_uint4(pack_bf16(g[8 * ch] * ca, g[8 * ch + 1] * ca), pack_bf16(g[8 * ch + 2] * ca, g[8 * ch + 3] * ca),
                                   pack_bf16(g[8 * ch + 4] * cb, g[8 * ch + 5] * cb), pack_bf16(g[8 * ch + 6] * cb, g[8 * ch + 7] * cb));
              *reinterpret_cast<uint4*>(&sg.q[(jc >> 1) * TILE + swz(row, 32 * (jc & 1) + 8 * ch)]) = u;
            }
          }
        }
        {  // S *= exp(da_cs of the last token)
          const float dec = __expf(sg.dcs[nv - 1]);
#pragma unroll
          for (int ph = 0; ph < 2; ++ph) {
            float v[32];
            tmem_ld32(tl + TM_S + 32 * ph, v);
            tmem_wait_ld();
#pragma unroll
            for (int p = 0; p < 32; ++p) v[p] *= dec;
            tmem_st32(tl + TM_S + 32 * ph, v);
          }
          tmem_wait_st();
        }
        fence_async();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&S.gm_ready);
        mbar_wait(&S.o_done[b], (gi >> 1) & 1);
        tc_fence_after();
        if (it + 1 < nsb) {  // bf16 copy of the updated state for the next sub-block
#pragma unroll
          for (int ph = 0; ph < 2; ++ph) {
            float v[32];
            tmem_ld32(tl + TM_S + 32 * ph, v);
            tmem_wait_ld();
#pragma unroll
            for (int ch = 0; ch < 4; ++ch) {
              uint4 u = make_uint4(pack_bf16(v[8 * ch], v[8 * ch + 1]), pack_bf16(v[8 * ch + 2], v[8 * ch + 3]),
                                   pack_bf16(v[8 * ch + 4], v[8 * ch + 5]), pack_bf16(v[8 * ch + 6], v[8 * ch + 7]));
              *reinterpret_cast<uint4*>(&S.sb[swz(row, 32 * ph + 8 * ch)]) = u;
            }
          }
          fence_async();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&S.sb_ready);
        }
      }
    }
    }
  } else {
    {
    // ================= epilogue B (rows 32 (w % 4) + lane, column half hf): outputs =================
    const int sp = warp & 3;
    const int row = 32 * sp + lane, tok = row >> 2, rank = row & 3;
    const uint32_t tl = tmem + ((uint32_t)(32 * sp) << 16);
    int gi = 0;
    for (int item = blockIdx.x; item < nitems; item += gridDim.x) {
      const int c = item / H, h = item % H;
      const int cs = ch_start[c], nsb = (ch_len[c] + C - 1) / C;
      for (int it = 0; it < nsb; ++it, ++gi) {
        const int s = gi & 1, b = gi & 1, t0 = cs + it * C;
        Stage& sg = S.stage[s];
        mbar_wait(&S.full[s], (gi >> 1) & 1);
        const int nv = sg.nv;
        mbar_wait(&S.o_done[b], (gi >> 1) & 1);
        tc_fence_after();
        const uint32_t tob = tl + 128 * b;
#pragma unroll 1
        for (int hf = 0; hf < 2; ++hf) {
          const float er = __expf(sg.dcs[tok]), dsk = sg.dsk;
          float o1[32], o2[32];
          tmem_ld32(tob + TM_O1 + 32 * hf, o1);
          tmem_ld32(tob + TM_O2 + 32 * hf, o2);
          tmem_wait_ld();
          const float* phr = &sg.phi[rank * PP + 32 * hf];
          const float* zer = &sg.zeta[rank * PP + 32 * hf];
#pragma unroll
          for (int ch = 0; ch < 4; ++ch) {
            const int p0 = 32 * hf + 8 * ch;
            U8 pv, zz;
            pv.u = *reinterpret_cast<const uint4*>(&sg.pv[swz(row, p0)]);
            zz.u = *reinterpret_cast<const uint4*>(&sg.z[swz(tok, p0)]);
            const float4 pa = *reinterpret_cast<const float4*>(phr + 8 * ch), pb = *reinterpret_cast<const float4*>(phr + 8 * ch + 4);
            const float4 za = *reinterpret_cast<const float4*>(zer + 8 * ch), zb = *reinterpret_cast<const float4*>(zer + 8 * ch + 4);
            const float ph8[8] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
            const float ze8[8] = {za.x, za.y, za.z, za.w, zb.x, zb.y, zb.z, zb.w};
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              const float g0 = __bfloat162float(zz.h[e]) * ze8[e];
              const float gz = ph8[e] * rbf(fmaf(g0, tanh_approx(g0), g0));
              o1[8 * ch + e] = (er * o1[8 * ch + e] + o2[8 * ch + e] + dsk * __bfloat162float(pv.h[e])) * gz;
            }
          }
          // rank sum by a transposing butterfly: after the two steps lane rank r holds the full sums of columns
          // 16 (r & 1) + 8 (r >> 1) .. + 8
          const bool b0 = rank & 1, b1 = rank & 2;
          float hsum[16];
#pragma unroll
          for (int e = 0; e < 16; ++e) {
            const float send = b0 ? o1[e] : o1[16 + e];
            hsum[e] = (b0 ? o1[16 + e] : o1[e]) + __shfl_xor_sync(0xffffffffu, send, 1);
          }
          float qs[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float send = b1 ? hsum[e] : hsum[8 + e];
            qs[e] = (b1 ? hsum[8 + e] : hsum[e]) + __shfl_xor_sync(0xffffffffu, send, 2);
          }
          if (tok < nv)
            *reinterpret_cast<uint4*>(out + ((long)(t0 + tok) * H + h) * P + 32 * hf + 16 * (rank & 1) + 8 * (rank >> 1)) =
                make_uint4(pack_bf16(qs[0], qs[1]), pack_bf16(qs[2], qs[3]), pack_bf16(qs[4], qs[5]), pack_bf16(qs[6], qs[7]));
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) { mbar_arrive(&S.o_free[b]); mbar_arrive(&S.empty[s]); }
      }
    }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == MMA_WARP) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tmem), "n"(TMEM_COLS));
  }
}

// ======================================================================================================
// Chunk-local states (replaces _ssd_chunk_state on sm_100 / sm_103): per 128-token chunk and head
//   Sloc[n][p] = sum_j khat_j^T (PsiV_j * ts_j * exp(dtot - dacs_j))      (fp32, layout (NC, H, N, P))
// with khat rotated by the chunk-local phase thl (the state pass applies the chunk's phase offset), as the sm90 kernel.
// Producers (TMA raw K / V / phases into a three-deep ring, in-place bias + rotation, weighted PsiV), one MMA thread
// (UMMA M=128 N=64 per 32-token sub-block into one of two TMEM accumulators), four epilogue warps (TMEM -> global).
namespace cs {
constexpr int NPROD = 11, MMA_WARP = 4, PROD0 = 5, THREADS = (PROD0 + NPROD) * 32, PT = NPROD * 32, NST = 3;
constexpr int MAXI = 512;  // per-CTA work items whose chunk layout is staged in smem (later ones read global)
constexpr unsigned TX = 2 * TILE_B + C * P * 2 + C * NQ * 4;
struct __align__(1024) Stage {
  __nv_bfloat16 k[2 * TILE];   // K -> khat [n half][row][64] (A, MN-major)
  __nv_bfloat16 pv[TILE];      // PsiV * w [row][p] (B, MN-major)
  __nv_bfloat16 v[C * P];      // raw V (SW128)
  float th[C * NQ], sn[C * NQ];
  float dacs[C], ts[C];
};
struct __align__(16) Consts { float kb[R * N], psi[R * P], dl[4]; };  // one head of one chunk, prefetched one item ahead
struct Smem {
  Stage stage[NST];
  Consts cst[2];
  int m_start[MAXI], m_len[MAXI];
  unsigned long long tfull[NST], full[NST], empty[NST], done[2], efree[2], cfull[2];
  uint32_t tmem_base;
};
struct Maps { CUtensorMap k, v, th; };

__device__ __forceinline__ void cp_arrive(void* mbar) {  // the mbarrier gets this thread's arrival once its cp.async land
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(dst)), "l"(src));
}

__global__ void __launch_bounds__(THREADS, 1) chunk_state_tc_kernel(
    const __grid_constant__ Maps maps, const float* __restrict__ kbias, const float* __restrict__ psi,
    const float* __restrict__ dacs, const float* __restrict__ ts_g, const float* __restrict__ dtot,
    const int* __restrict__ ch_start, const int* __restrict__ ch_len, float* __restrict__ sloc, int T, int H, int NC) {
  extern __shared__ unsigned char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023));
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nitems = NC * H;
  if (blockIdx.x >= nitems) return;
  const int nmine = (nitems - blockIdx.x + gridDim.x - 1) / gridDim.x;
  if (warp == MMA_WARP) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(smem_u32(&S.tmem_base)), "n"(128));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
  }
  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(&S.tfull[i], 1 + 32); mbar_init(&S.full[i], NPROD); mbar_init(&S.empty[i], 1); }
    for (int i = 0; i < 2; ++i) { mbar_init(&S.done[i], 1); mbar_init(&S.efree[i], 4); mbar_init(&S.cfull[i], PT); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  for (int k = tid; k < min(nmine, MAXI); k += THREADS) {
    const int c = (blockIdx.x + k * gridDim.x) / H;
    S.m_start[k] = ch_start[c];
    S.m_len[k] = ch_len[c];
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = S.tmem_base;
  auto meta = [&](int k, int& st, int& len) {  // chunk start / length of this CTA's k-th item
    if (k < MAXI) { st = S.m_start[k]; len = S.m_len[k]; }
    else { const int c = (blockIdx.x + k * gridDim.x) / H; st = ch_start[c]; len = ch_len[c]; }
  };

  if (warp >= PROD0) {
    const int pt = tid - PROD0 * 32;
    const int cr = pt & 3, rr = (pt >> 3) & 3, cA = 8 * cr, cB = 32 + 8 * cr;
    auto unit_row = [](int u) { return (u >> 5) * 8 + ((u >> 2) & 1) * 4 + ((u >> 3) & 3); };
    auto fetch_consts = [&](int k) {  // item k's head constants into cst[k & 1] (cp.async group)
      if (k < nmine) {
        const int item = blockIdx.x + k * gridDim.x, c = item / H, h = item % H;
        Consts& cb = S.cst[k & 1];
        for (int j = pt; j < (R * N + R * P) / 4; j += PT) {
          if (j < R * N / 4) cp16(&cb.kb[4 * j], kbias + h * R * N + 4 * j);
          else cp16(&cb.psi[4 * (j - R * N / 4)], psi + h * R * P + 4 * (j - R * N / 4));
        }
        if (pt == 0) cp4(&cb.dl[0], dtot + c * H + h, true);
        cp_arrive(&S.cfull[k & 1]);
      }
    };
    // raw inputs of sub-block gi + NST - 1 go out once sub-block gi is handed over (cursor la)
    int la_k = 0, la_it = 0, la_gi = 0;
    auto issue = [&]() {
      if (la_k >= nmine) return;
      int lcs, lcl;
      meta(la_k, lcs, lcl);
      const int lh = (blockIdx.x + la_k * gridDim.x) % H, lt0 = lcs + la_it * C, lnv = min(C, lcl - la_it * C);
      const int s = la_gi % NST;
      Stage& sg = S.stage[s];
      if (warp == PROD0) {
        if (la_gi >= NST) mbar_wait(&S.empty[s], ((la_gi / NST) + 1) & 1);
        if (lane == 0) {
          mbar_arrive_expect_tx(&S.tfull[s], TX);
          tma2d(&sg.k[0], &maps.k, 0, lt0 * R, &S.tfull[s]);
          tma2d(&sg.k[TILE], &maps.k, 64, lt0 * R, &S.tfull[s]);
          tma3d(sg.v, &maps.v, 0, lh, lt0, &S.tfull[s]);
          tma3d(sg.th, &maps.th, 0, lh, lt0, &S.tfull[s]);
        }
        const bool ok = lane < lnv;
        cp4(sg.dacs + lane, dacs + (ok ? (long)lh * T + lt0 + lane : 0), ok);
        cp4(sg.ts + lane, ts_g + (ok ? (long)lh * T + lt0 + lane : 0), ok);
        cp_arrive(&S.tfull[s]);
      }
      ++la_gi;
      if (++la_it == (lcl + C - 1) / C) { la_it = 0; ++la_k; }
    };
    fetch_consts(0);
    issue();
    issue();
    int gi = 0;
    for (int k = 0; k < nmine; ++k) {
      int cs0, cl;
      meta(k, cs0, cl);
      const int nsb = (cl + C - 1) / C;
      Consts& cb = S.cst[k & 1];
      mbar_wait(&S.cfull[k & 1], (k >> 1) & 1);
      bar_sync(BAR_PROD, PT);  // item k's constants (and the raw scalars issued so far) visible; cst[(k + 1) & 1] free
      fetch_consts(k + 1);
      const float dl = cb.dl[0];
      U8 bk[4];
      {
        const float* kr = &cb.kb[rr * N];
        const int cols[4] = {cA, cA + 64, cB, cB + 64};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float4 x = *reinterpret_cast<const float4*>(kr + cols[q]), y = *reinterpret_cast<const float4*>(kr + cols[q] + 4);
          bk[q].h2[0] = __floats2bfloat162_rn(x.x, x.y); bk[q].h2[1] = __floats2bfloat162_rn(x.z, x.w);
          bk[q].h2[2] = __floats2bfloat162_rn(y.x, y.y); bk[q].h2[3] = __floats2bfloat162_rn(y.z, y.w);
        }
      }
      for (int it = 0; it < nsb; ++it, ++gi) {
        const int s = gi % NST, nv = min(C, cl - it * C);
        Stage& sg = S.stage[s];
        mbar_wait(&S.tfull[s], (gi / NST) & 1);
        bar_sync(BAR_PROD, PT);
        for (int idx = pt; idx < C * NQ; idx += PT) {
          float th = sg.th[idx];
          th = th - 6.283185307179586f * floorf(th * 0.15915494309189535f);
          float sv, cv;
          __sincosf(th, &sv, &cv);
          sg.th[idx] = cv;
          sg.sn[idx] = sv;
        }
        for (int u = pt; u < ROWS * 8; u += PT) {  // PsiV * w, (row, 16-byte piece)
          const int row = u >> 3, ci = u & 7, tok = row >> 2, r4 = row & 3;
          const float wt = tok < nv ? sg.ts[tok] * __expf(dl - sg.dacs[tok]) : 0.f;
          U8 va;
          va.u = *reinterpret_cast<const uint4*>(&sg.v[swz(tok, 8 * ci)]);
          const float4 pa = *reinterpret_cast<const float4*>(&cb.psi[r4 * P + 8 * ci]);
          const float4 pb = *reinterpret_cast<const float4*>(&cb.psi[r4 * P + 8 * ci + 4]);
          const float ps[8] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 x = __bfloat1622float2(va.h2[j]);
            va.h2[j] = __floats2bfloat162_rn(x.x * rbf(ps[2 * j]) * wt, x.y * rbf(ps[2 * j + 1]) * wt);
          }
          *reinterpret_cast<uint4*>(&sg.pv[swz(row, 8 * ci)]) = va.u;
        }
        bar_sync(BAR_PROD, PT);
#pragma unroll 1
        for (int u = pt; u < ROWS * 4; u += PT) {
          const int row = unit_row(u), tok = row >> 2;
          const int oA = swz(row, cA), oB = swz(row, cB);
          U8 k4[4];
          k4[0].u = *reinterpret_cast<const uint4*>(&sg.k[oA]);
          k4[1].u = *reinterpret_cast<const uint4*>(&sg.k[TILE + oA]);
          k4[2].u = *reinterpret_cast<const uint4*>(&sg.k[oB]);
          k4[3].u = *reinterpret_cast<const uint4*>(&sg.k[TILE + oB]);
          const float4* cp = reinterpret_cast<const float4*>(&sg.th[tok * NQ + cA]);
          const float4* sp = reinterpret_cast<const float4*>(&sg.sn[tok * NQ + cA]);
          const float4 ca = cp[0], cb4 = cp[1], sa = sp[0], sb = sp[1];
          const float cn[8] = {ca.x, ca.y, ca.z, ca.w, cb4.x, cb4.y, cb4.z, cb4.w};
          const float snv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
#pragma unroll
          for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int j = 0; j < 4; ++j) k4[q].h2[j] = __hadd2(k4[q].h2[j], bk[q].h2[j]);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float c_a = cn[2 * j], c_b = cn[2 * j + 1], s_a = snv[2 * j], s_b = snv[2 * j + 1];
            const float2 y0 = __bfloat1622float2(k4[0].h2[j]), y1 = __bfloat1622float2(k4[1].h2[j]);
            k4[0].h2[j] = __floats2bfloat162_rn(c_a * y0.x - s_a * y1.x, c_b * y0.y - s_b * y1.y);
            k4[1].h2[j] = __floats2bfloat162_rn(s_a * y0.x + c_a * y1.x, s_b * y0.y + c_b * y1.y);
          }
          *reinterpret_cast<uint4*>(&sg.k[oA]) = k4[0].u;
          *reinterpret_cast<uint4*>(&sg.k[TILE + oA]) = k4[1].u;
          *reinterpret_cast<uint4*>(&sg.k[oB]) = k4[2].u;
          *reinterpret_cast<uint4*>(&sg.k[TILE + oB]) = k4[3].u;
        }
        fence_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&S.full[s]);
        issue();
      }
    }
  } else if (warp == MMA_WARP) {
    if (lane == 0) {
      constexpr uint32_t ID_S = idesc(128, 64, 1, 1);
      int gi = 0;
      for (int k = 0; k < nmine; ++k) {
        int cs0, cl;
        meta(k, cs0, cl);
        const int nsb = (cl + C - 1) / C, b = k & 1;
        if (k >= 2) mbar_wait(&S.efree[b], ((k >> 1) + 1) & 1);
        for (int it = 0; it < nsb; ++it, ++gi) {
          const int s = gi % NST;
          Stage& sg = S.stage[s];
          mbar_wait(&S.full[s], (gi / NST) & 1);
          tc_fence_after();
#pragma unroll
          for (int kk = 0; kk < 8; ++kk)
            umma(tmem + 64 * b, mdesc(&sg.k[kk * 16 * 64]), mdesc(&sg.pv[kk * 16 * 64]), ID_S, (it | kk) > 0);
          umma_commit(&S.empty[s]);
        }
        umma_commit(&S.done[b]);
      }
    }
    __syncwarp();
  } else {
    const int row = 32 * warp + lane;
    const uint32_t tl = tmem + ((uint32_t)(32 * warp) << 16);
    for (int k = 0; k < nmine; ++k) {
      const int b = k & 1, item = blockIdx.x + k * gridDim.x;
      mbar_wait(&S.done[b], (k >> 1) & 1);
      tc_fence_after();
      float* dst = sloc + (long)item * N * P + (long)row * P;  // item = c * H + h
#pragma unroll
      for (int ph = 0; ph < 2; ++ph) {
        float v[32];
        tmem_ld32(tl + 64 * b + 32 * ph, v);
        tmem_wait_ld();
#pragma unroll
        for (int q = 0; q < 8; ++q)
          *reinterpret_cast<float4*>(dst + 32 * ph + 4 * q) = make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.efree[b]);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == MMA_WARP) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tmem), "n"(128));
  }
}
}  // namespace cs
}  // namespace

// ---------------------------------------------------------------- host
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn get_encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qres;
    void* p = nullptr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qres);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<EncodeTiledFn>(p);
  }
  return fn;
}
static CUtensorMap make_map(const void* base, CUtensorMapDataType dt, int rank, const cuuint64_t* dims, const cuuint64_t* strides,
                            const cuuint32_t* box, CUtensorMapSwizzle sw) {
  CUtensorMap m;
  const cuuint32_t estr[3] = {1, 1, 1};
  const CUresult r = get_encode_fn()(&m, dt, rank, const_cast<void*>(base), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

void mamba3_scan_prefill_tc(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Z,
                            torch::Tensor qbias, torch::Tensor kbias, torch::Tensor psi, torch::Tensor phi, torch::Tensor zeta,
                            torch::Tensor Dskip, torch::Tensor thl, torch::Tensor ain, torch::Tensor dcs32, torch::Tensor dcsr32,
                            torch::Tensor ts, torch::Tensor gam, torch::Tensor ch_start, torch::Tensor ch_len, torch::Tensor ch_last,
                            torch::Tensor sin_t, torch::Tensor out, torch::Tensor fin_k, int64_t NC) {
  const int T = V.size(0), H = V.size(1);
  TORCH_CHECK(V.size(2) == P && Q.size(-1) == N && Q.is_contiguous() && K.is_contiguous() && V.is_contiguous() && Z.is_contiguous());
  TORCH_CHECK(out.is_contiguous() && sin_t.is_contiguous() && thl.is_contiguous() && thl.size(-1) == NQ);
  Maps M;
  {
    const cuuint64_t d[2] = {(cuuint64_t)N, (cuuint64_t)T * R}, st[1] = {(cuuint64_t)N * 2};
    const cuuint32_t box[2] = {64, ROWS};
    M.q = make_map(Q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d, st, box, CU_TENSOR_MAP_SWIZZLE_128B);
    M.k = make_map(K.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d, st, box, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  {
    const cuuint64_t d[3] = {(cuuint64_t)P, (cuuint64_t)H, (cuuint64_t)T}, st[2] = {(cuuint64_t)P * 2, (cuuint64_t)H * P * 2};
    const cuuint32_t box[3] = {P, 1, C};
    M.v = make_map(V.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d, st, box, CU_TENSOR_MAP_SWIZZLE_128B);
    M.z = make_map(Z.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d, st, box, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  {
    const cuuint64_t d[3] = {(cuuint64_t)NQ, (cuuint64_t)H, (cuuint64_t)T}, st[2] = {(cuuint64_t)NQ * 4, (cuuint64_t)H * NQ * 4};
    const cuuint32_t box[3] = {NQ, 1, C};
    M.th = make_map(thl.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, d, st, box, CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  const int smem = (int)sizeof(Smem) + 1024;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(mamba3_scan_prefill_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  static int nsm = 0;
  if (!nsm) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, out.get_device());
  const int grid = (int)std::min<int64_t>(NC * H, nsm);
  mamba3_scan_prefill_tc_kernel<<<grid, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      M, qbias.data_ptr<float>(), kbias.data_ptr<float>(), psi.data_ptr<float>(), phi.data_ptr<float>(), zeta.data_ptr<float>(),
      Dskip.data_ptr<float>(), ain.data_ptr<float>(), dcs32.data_ptr<float>(), dcsr32.data_ptr<float>(),
      ts.data_ptr<float>(), gam.data_ptr<float>(), ch_start.data_ptr<int>(), ch_len.data_ptr<int>(), ch_last.data_ptr<int>(),
      sin_t.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<__nv_bfloat16*>(fin_k.data_ptr()),
      T, H, (int)NC);
}

void mamba3_chunk_state_tc(torch::Tensor K, torch::Tensor V, torch::Tensor kbias, torch::Tensor psi, torch::Tensor thl,
                           torch::Tensor dacs, torch::Tensor ts, torch::Tensor dtot, torch::Tensor ch_start, torch::Tensor ch_len,
                           torch::Tensor sloc, int64_t NC) {
  const int T = V.size(0), H = V.size(1);
  TORCH_CHECK(V.size(2) == P && K.size(-1) == N && K.is_contiguous() && V.is_contiguous() && thl.is_contiguous() && sloc.is_contiguous());
  cs::Maps M;
  {
    const cuuint64_t d[2] = {(cuuint64_t)N, (cuuint64_t)T * R}, st[1] = {(cuuint64_t)N * 2};
    const cuuint32_t box[2] = {64, ROWS};
    M.k = make_map(K.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d, st, box, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  {
    const cuuint64_t d[3] = {(cuuint64_t)P, (cuuint64_t)H, (cuuint64_t)T}, st[2] = {(cuuint64_t)P * 2, (cuuint64_t)H * P * 2};
    const cuuint32_t box[3] = {P, 1, C};
    M.v = make_map(V.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d, st, box, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  {
    const cuuint64_t d[3] = {(cuuint64_t)NQ, (cuuint64_t)H, (cuuint64_t)T}, st[2] = {(cuuint64_t)NQ * 4, (cuuint64_t)H * NQ * 4};
    const cuuint32_t box[3] = {NQ, 1, C};
    M.th = make_map(thl.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, d, st, box, CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  const int smem = (int)sizeof(cs::Smem) + 1024;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(cs::chunk_state_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  static int nsm = 0;
  if (!nsm) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, sloc.get_device());
  const int grid = (int)std::min<int64_t>(NC * H, nsm);
  cs::chunk_state_tc_kernel<<<grid, cs::THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      M, kbias.data_ptr<float>(), psi.data_ptr<float>(), dacs.data_ptr<float>(), ts.data_ptr<float>(), dtot.data_ptr<float>(),
      ch_start.data_ptr<int>(), ch_len.data_ptr<int>(), sloc.data_ptr<float>(), T, H, (int)NC);
}
int64_t mamba3_scan_prefill_tc_smem() { return sizeof(Smem) + 1024; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scan", &mamba3_scan_prefill_tc);
  m.def("chunk_state", &mamba3_chunk_state_tc);
  m.def("smem", &mamba3_scan_prefill_tc_smem);
}
