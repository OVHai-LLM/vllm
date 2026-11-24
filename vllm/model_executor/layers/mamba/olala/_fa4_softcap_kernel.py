# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4 sm100 forward with a native softcap; see ``fa4_softcap.py``.

``softmax_step`` is ``FlashAttentionForwardSm100.softmax_step`` (sha256[:16] 47426257b8efc9dd)
with its score_mod / mask / row-max block replaced. Only used with the marker score_mods of
``fa4_softcap`` (``softcap_score_mod``, or ``softcap_prescaled_score_mod`` when q already
carries ``scale / softcap`` and the multiply is skipped).
"""
# ruff: noqa: F403, F405
from vllm.vllm_flash_attn.cute.flash_fwd_sm100 import *
from vllm.vllm_flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100


class OlalaSoftcapFwdSm100(FlashAttentionForwardSm100):
    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        # Native softcap (see fa4_softcap.py): mask the raw scores, take the row max on them
        # (tanh is monotonic), then tanh(s * scale / cap); exp2 uses cap * log2e.
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        # prescaled: q already carries scale / cap, the scores are the tanh arguments
        prescaled = getattr(self.score_mod, "__olala_prescaled__", False)
        k_scale = softmax.softmax_scale
        max_raw = utils.fmax_reduce(tSrS_t2r.load(), None, arch=softmax.arch)
        if const_expr(prescaled):
            row_max_new = cute.math.tanh(max_raw, fastmath=True)
        else:
            row_max_new = cute.math.tanh(max_raw * k_scale, fastmath=True)
        if const_expr(is_first):
            row_max_new = row_max_new if max_raw != -Float32.inf else -Float32.inf
        else:
            row_max_new = cute.arch.fmax(row_max_new, softmax.row_max[0])
        if const_expr(mask_fn is not None):
            for i in cutlass.range_constexpr(cute.size(tSrS_t2r.shape)):
                s = tSrS_t2r[i]
                if const_expr(not prescaled):
                    s = s * k_scale
                tSrS_t2r[i] = cute.math.tanh(s, fastmath=True) if s != -Float32.inf else -Float32.inf
        elif const_expr(prescaled):
            for i in cutlass.range_constexpr(cute.size(tSrS_t2r.shape)):
                tSrS_t2r[i] = cute.math.tanh(tSrS_t2r[i], fastmath=True)
        else:
            # packed FMUL2 for the scale (same rounding as a scalar multiply)
            for i in cutlass.range_constexpr(0, cute.size(tSrS_t2r.shape), 2):
                a, b = cute.arch.mul_packed_f32x2((tSrS_t2r[i], tSrS_t2r[i + 1]), (k_scale, k_scale))
                tSrS_t2r[i] = cute.math.tanh(a, fastmath=True)
                tSrS_t2r[i + 1] = cute.math.tanh(b, fastmath=True)
        row_max, acc_scale = softmax.update_row_max_from_local(row_max_new, is_first)

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape, Float32
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        )
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.n_block_size
                if const_expr(i + 1 == split_P_arrive_idx):
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase ^ 1, s0_s1_sequence_phase ^ 1
