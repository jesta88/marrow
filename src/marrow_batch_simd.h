/* marrow batch SIMD kernel - across-instance SoA, vertical `wide` math.
 *
 * ONE source. Each ISA translation unit (marrow_batch_{sse2,avx2,avx2_fma}.c) defines its
 * MRW_ISA_* selector and MRW_KERNEL (the exported function name), then includes this file.
 *
 * The per-joint pose pipeline - nlerp, TRS→affine, affine compose - is *purely vertical*:
 * each TRS/matrix component is one `wide` register holding MRW_W instances, so there are no
 * cross-128-lane ops and AVX2 is genuinely SSE2's structure at 8-wide. The only lane-crossing
 * steps are the per-lane keyframe gather (lane l reads frame i0[l]) and the SoA→AoS palette
 * scatter; those are specialized PER ISA (AVX2 hardware vgather + an in-register transpose; SSE2 a
 * scalar deinterleave) below, off the `wide` path, without touching the vertical math.
 *
 * Inputs are pre-validated by the dispatcher in marrow_batch.c; `model` is the validated SoA
 * scratch (joint_count × 12 × MRW_LANES floats). Result is bit-for-bit-close to the scalar
 * reference (differences are FMA fusion / reassociation - the variation is visual-only). */
#include "marrow_batch_kernel.h"

#ifndef MRW_KERNEL
#  error "define MRW_KERNEL (the kernel function name) before including marrow_batch_simd.h"
#endif

mrw_result MRW_KERNEL(const mrw_skeleton_view *skel, const mrw_clip_view *clip,
                      const float *times, uint32_t instance_count,
                      void *out_palettes, mrw_palette_format out_format,
                      int use_f16c, float *model) {
    const uint32_t jc  = skel->joint_count;
    const uint32_t SUB = MRW_LANES / MRW_W; /* per-tile subgroups: 1 (AVX2) or 2 (SSE2) */
    /* codec is per-CALL constant (loop-invariant): branch on it at the gather/affine SEAMS so the
     * stride/index math stays constant-foldable within each arm. codec 0 = 10 f32/sample (q4+t3+s3,
     * 40 B); codec 1 = 7 f32/sample (q4+t3, 28 B) with scale ≡ (1,1,1) - drops 6 gathers + 9 muls. */
    const uint32_t codec = clip->codec;
    const uint32_t scw   = codec ? 7u : 10u; /* sample float-units (per-sample stride in floats) */
    (void)scw; /* used only by the AVX2 vgather path; SSE2/scalar gathers use the stride constants */

    /* Clip sample run base (validated). The gather reads samples here directly instead of routing
     * every lane through the external mrw_clip_sample (joint j, sample s starts at sbase + (j*sample
     * _count + s)*10 floats / sbytes + …*40 bytes). */
#if defined(MRW_ISA_AVX2) && !defined(MRW_AVX2_GATHER_SCALAR)
    const mrw_blob_f32 *sbase = (const mrw_blob_f32 *)(const void *)(clip->base + clip->samples_off);
#else
    const uint8_t *sbytes = clip->base + clip->samples_off;
#endif

    for (uint32_t base = 0; base < instance_count; ) {
        uint32_t live = instance_count - base;
        if (live > MRW_LANES) live = MRW_LANES;

        /* frame state. `st` is clip-global (sample_count==1 or duration==0) - identical for
         * every lane - so the static/dynamic branch never diverges across a tile. Padded lanes
         * [live, MRW_LANES) take a SAFE in-range time (times[base]); they are computed, never scattered. */
        uint32_t i0[MRW_LANES]; float uu[MRW_LANES]; int st = 0;
        for (uint32_t l = 0; l < MRW_LANES; ++l) {
            float t = times[base + (l < live ? l : 0)];
            uint32_t fi = 0; float fu = 0.0f;
            st = mrw_clip_frame(clip, t, &fi, &fu);
            i0[l] = fi; uu[l] = fu;
        }
#if defined(MRW_ISA_AVX2) && !defined(MRW_AVX2_GATHER_SCALAR)
        /* per-lane float-unit gather index i0[l]*scw (10 for codec 0, 7 for codec 1), constant across
         * joints in this tile. */
        __m256i gidx = _mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(const void *)i0),
                                          _mm256_set1_epi32((int)scw));
#endif

        for (uint32_t j = 0; j < jc; ++j) {
            int is_root = (j == 0);
            /* parent + inverse_bind are per-joint and constant across tiles/lanes; read them inline
             * from the validated skeleton - no per-tile call into mrw_skeleton_*. parent < j. */
            uint16_t parent = is_root ? 0
                : mrw_rd_u16(skel->base + skel->parent_off + (size_t)j * MRW_PARENT_STRIDE);

            /* inverse_bind is shared by every lane → broadcast once and fold M_j = model(j) ∘
             * inverse_bind in the VERTICAL path (one wide affine_mul per subgroup), not a scalar
             * mrw_affine_mul per lane. That scalar per-lane compose was the dominant ISA-independent
             * cost - folding it wide is what lets AVX2's 8-wide actually pay off over SSE2. */
            float ib[MRW_PALETTE_FLOATS];
            memcpy(ib, skel->base + skel->inverse_bind_off + (size_t)j * MRW_INVERSE_BIND_STRIDE,
                   MRW_INVERSE_BIND_STRIDE);
            mrw_w ibw[12];
            for (int c = 0; c < 12; ++c) ibw[c] = mrw_w_set1(ib[c]);
#if !defined(MRW_ISA_AVX2)
            _Alignas(MRW_W_ALIGN) float pal_soa[MRW_PALETTE_FLOATS * MRW_LANES];
#endif

            for (uint32_t g = 0; g < SUB; ++g) {
                uint32_t L = g * MRW_W; /* first lane of this subgroup */
                mrw_w la[12];

                if (st) {
                    /* static clip: pose is sample 0 for all lanes → broadcast (no gather). */
                    mrw_trs s0; mrw_clip_sample(clip, j, 0, &s0);
                    float aff[12]; mrw_trs_to_affine(&s0, aff);
                    for (int c = 0; c < 12; ++c) la[c] = mrw_w_set1(aff[c]);
                } else {
                    /* gather the two bracket keyframes per ISA (vgather on AVX2, inline read on SSE2),
                     * then run the shared vertical interp: nlerp the quat, lerp trans (+scale for codec 0).
                     * The codec branch is loop-invariant (per call) → the stride math constant-folds. */
                    mrw_w A_[10], B_[10];
#if defined(MRW_ISA_AVX2) && !defined(MRW_AVX2_GATHER_SCALAR)
                    if (codec) mrw_gather_vg_c1(sbase + (size_t)j * clip->sample_count * scw, gidx, A_, B_);
                    else       mrw_gather_vg   (sbase + (size_t)j * clip->sample_count * scw, gidx, A_, B_);
#else
                    if (codec) mrw_gather_sc_c1(sbytes + (size_t)j * clip->sample_count * MRW_ROOT_SAMPLE_STRIDE,
                                                i0, L, A_, B_);
                    else       mrw_gather_sc   (sbytes + (size_t)j * clip->sample_count * MRW_CLIP_SAMPLE_STRIDE,
                                                i0, L, A_, B_);
#endif
                    mrw_w Uv = mrw_w_loadu(&uu[L]);
                    mrw_w qx, qy, qz, qw;
                    mrw_quat_nlerp_w(A_[0], A_[1], A_[2], A_[3], B_[0], B_[1], B_[2], B_[3],
                                     Uv, &qx, &qy, &qz, &qw);
                    mrw_w tx = mrw_w_fma(Uv, mrw_w_sub(B_[4], A_[4]), A_[4]); /* lerp t */
                    mrw_w ty = mrw_w_fma(Uv, mrw_w_sub(B_[5], A_[5]), A_[5]);
                    mrw_w tz = mrw_w_fma(Uv, mrw_w_sub(B_[6], A_[6]), A_[6]);
                    if (codec) {
                        /* codec 1: scale ≡ 1 → skip the 3 scale lerps and the scale-free affine drops 9 muls. */
                        mrw_trs_to_affine_w_ns(qx, qy, qz, qw, tx, ty, tz, la);
                    } else {
                        mrw_w sx = mrw_w_fma(Uv, mrw_w_sub(B_[7], A_[7]), A_[7]); /* lerp s */
                        mrw_w sy = mrw_w_fma(Uv, mrw_w_sub(B_[8], A_[8]), A_[8]);
                        mrw_w sz = mrw_w_fma(Uv, mrw_w_sub(B_[9], A_[9]), A_[9]);
                        mrw_trs_to_affine_w(qx, qy, qz, qw, tx, ty, tz, sx, sy, sz, la);
                    }
                }

                /* compose model(j): root → local; else parent_model ∘ local, all vertical. */
                mrw_w mj[12];
                if (is_root) {
                    for (int c = 0; c < 12; ++c) mj[c] = la[c];
                } else {
                    mrw_w pm[12];
                    for (int c = 0; c < 12; ++c) pm[c] = mrw_w_load(&MRW_MODEL_AT(model, parent, c, L));
                    mrw_affine_mul_w(pm, la, mj);
                }
                for (int c = 0; c < 12; ++c) mrw_w_store(&MRW_MODEL_AT(model, j, c, L), mj[c]);

                /* M_j = model(j) ∘ inverse_bind, VERTICAL → palette. */
                mrw_w Mj[12];
                mrw_affine_mul_w(mj, ibw, Mj);
#if defined(MRW_ISA_AVX2)
#  if defined(MRW_BENCH_NO_SCATTER)
                /* compute-only floor: keep the fold live, drop the palette write (no DRAM). */
                (void)out_palettes; (void)out_format; (void)use_f16c; (void)base; (void)jc; (void)live;
                mrw_bench_consume(Mj);
#  else
                /* AVX2 (one subgroup): transpose the 8 lanes straight to the AoS output (f32 or f16). */
                mrw_scatter_palette(Mj, out_palettes, out_format, use_f16c, base, jc, j, live);
#  endif
#else
                for (int c = 0; c < 12; ++c) mrw_w_store(&pal_soa[(size_t)c * MRW_LANES + L], Mj[c]);
#endif
            }

#if !defined(MRW_ISA_AVX2)
            /* SSE2: scatter the SoA palette staging to AoS - LIVE lanes only (the SoA→AoS deinterleave;
             * the compose already happened wide above). Padded lanes are computed but never written.
             * SSE2 stays pure-SSE2 (no F16C in the 128-bit baseline TU): f16 narrows with the scalar
             * mrw_f32_to_f16 (parity-equal to vcvtps2ph for these finite values), use_f16c ignored. */
            (void)use_f16c;
            for (uint32_t l = 0; l < live; ++l) {
                size_t off = ((size_t)(base + l) * jc + j) * MRW_PALETTE_FLOATS;
                if (out_format == MRW_PALETTE_F16) {
                    uint16_t *dst = (uint16_t *)out_palettes + off;
                    for (int c = 0; c < 12; ++c) dst[c] = mrw_f32_to_f16(pal_soa[(size_t)c * MRW_LANES + l]);
                } else {
                    float *dst = (float *)out_palettes + off;
                    for (int c = 0; c < 12; ++c) dst[c] = pal_soa[(size_t)c * MRW_LANES + l];
                }
            }
#endif
        }

        base += live; /* overflow-safe: base only ever reaches instance_count */
    }
    return MRW_OK;
}
