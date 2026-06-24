/* marrow across-instance pose ops - fused 2-clip blend + additive-layer accumulate into the
 * batch palette (the crowd cross-fade / additive layer). Same contract as
 * mrw_batch_clip_to_palette: validated dispatch, caller-owned SoA scratch + _requirements()
 * sizing, f32 + f16 outputs, instance_count==0 no-op, overflow-checked. The SCALAR backend here
 * is the bit-for-bit reference (loops the single-pose blend/accumulate math over the instances);
 * the SSE2/AVX2(+FMA) kernels (marrow_blend_{sse2,avx2,avx2_fma}.c) match it within a small tolerance. */
#include "marrow_internal.h"
#include "marrow_quat.h"   /* mrw_quat_mul / mrw_quat_normalize (accumulate reference) */

static const float MRW_BB_IDENTITY[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

/* output element size; 0 ⇒ out-of-range format. */
static uint32_t bb_elem(mrw_palette_format f) {
    return f == MRW_PALETTE_F32 ? 4u : f == MRW_PALETTE_F16 ? 2u : 0u;
}

/* The scratch/output sizing equals the clip batch (the SoA model scratch and AoS palette are
 * identical; only the per-instance combine differs) - so the requirements DELEGATE to it. */
mrw_result mrw_batch_blend_clips_to_palette_requirements(
    uint32_t joint_count, uint32_t instance_count, mrw_palette_format out_format,
    mrw_mem_req *out_scratch, mrw_mem_req *out_palettes) {
    return mrw_batch_clip_to_palette_requirements(joint_count, instance_count, out_format,
                                                  out_scratch, out_palettes);
}
mrw_result mrw_batch_accumulate_to_palette_requirements(
    uint32_t joint_count, uint32_t instance_count, mrw_palette_format out_format,
    mrw_mem_req *out_scratch, mrw_mem_req *out_palettes) {
    return mrw_batch_clip_to_palette_requirements(joint_count, instance_count, out_format,
                                                  out_scratch, out_palettes);
}

/* ------------------------------------------------------------------ scalar reference */

/* Sample clip joint j at one lane's frame state into `lj` (matches mrw_clip_sample_local[j]). */
static void sample_joint(const mrw_clip_view *clip, uint32_t j, int st, uint32_t i0, float u, mrw_trs *lj) {
    if (st) {
        mrw_clip_sample(clip, j, 0, lj);
    } else {
        mrw_trs a, b;
        mrw_clip_sample(clip, j, i0, &a);
        mrw_clip_sample(clip, j, i0 + 1, &b);
        mrw_interp_trs(&a, &b, u, lj);
    }
}

/* Fold model(j) for one lane into the SoA scratch and (for live lanes) scatter the palette entry.
 * `local12` is the combined local affine; identical compose/scatter to batch_scalar. */
static void compose_scatter_scalar(float *model, const float ib[12],
                                   const float local12[12], int is_root, uint16_t parent,
                                   uint32_t j, uint32_t jc, uint32_t row, uint32_t l, int live_lane,
                                   void *out_palettes, mrw_palette_format out_format) {
    float mjm[12];
    if (is_root) {
        memcpy(mjm, local12, sizeof mjm);
    } else {
        float pm[12];
        for (uint32_t c = 0; c < MRW_PALETTE_FLOATS; ++c) pm[c] = MRW_MODEL_AT(model, parent, c, l);
        mrw_affine_mul(pm, local12, mjm);
    }
    for (uint32_t c = 0; c < MRW_PALETTE_FLOATS; ++c) MRW_MODEL_AT(model, j, c, l) = mjm[c];
    if (!live_lane) return;
    float M[12]; mrw_affine_mul(mjm, ib, M);
    size_t off = ((size_t)row * jc + j) * MRW_PALETTE_FLOATS;
    if (out_format == MRW_PALETTE_F16) {
        uint16_t *dst = (uint16_t *)out_palettes + off;
        for (uint32_t c = 0; c < MRW_PALETTE_FLOATS; ++c) dst[c] = mrw_f32_to_f16(M[c]);
    } else {
        memcpy((float *)out_palettes + off, M, sizeof M);
    }
}

static mrw_result blend_scalar(const mrw_skeleton_view *skel,
                               const mrw_clip_view *clipA, const mrw_clip_view *clipB,
                               const float *timesA, const float *timesB, const float *weights,
                               const float *mask, uint32_t instance_count,
                               void *out_palettes, mrw_palette_format out_format, float *model) {
    uint32_t jc = skel->joint_count;
    for (uint32_t base = 0; base < instance_count; ) {
        uint32_t live = instance_count - base; if (live > MRW_LANES) live = MRW_LANES;
        uint32_t i0A[MRW_LANES], i0B[MRW_LANES]; float uA[MRW_LANES], uB[MRW_LANES];
        int stA[MRW_LANES], stB[MRW_LANES];
        for (uint32_t l = 0; l < MRW_LANES; ++l) {
            uint32_t idx = base + (l < live ? l : 0);
            stA[l] = mrw_clip_frame(clipA, timesA[idx], &i0A[l], &uA[l]);
            stB[l] = mrw_clip_frame(clipB, timesB[idx], &i0B[l], &uB[l]);
        }
        for (uint32_t j = 0; j < jc; ++j) {
            int is_root = (j == 0);
            uint16_t parent = 0;
            if (!is_root) mrw_skeleton_parent(skel, j, &parent);
            float ib[12]; mrw_skeleton_inverse_bind(skel, j, ib);
            for (uint32_t l = 0; l < MRW_LANES; ++l) {
                uint32_t idx = base + (l < live ? l : 0);
                mrw_trs la, lb, blended;
                sample_joint(clipA, j, stA[l], i0A[l], uA[l], &la);
                sample_joint(clipB, j, stB[l], i0B[l], uB[l], &lb);
                float wc = mrw_clampf(weights[idx], 0.0f, 1.0f);
                float mj = mask ? mrw_clampf(mask[j], 0.0f, 1.0f) : 1.0f;
                float we = mrw_clampf(wc * mj, 0.0f, 1.0f);
                mrw_interp_trs(&la, &lb, we, &blended);            /* blend (== mrw_pose_blend) */
                float local12[12]; mrw_trs_to_affine(&blended, local12);
                compose_scatter_scalar(model, ib, local12, is_root, parent,
                                       j, jc, base + l, l, l < live, out_palettes, out_format);
            }
        }
        base += live;
    }
    return MRW_OK;
}

static mrw_result accum_scalar(const mrw_skeleton_view *skel, const mrw_clip_view *base_clip,
                               const float *times, const mrw_trs *delta, const float *weights,
                               const float *mask, uint32_t instance_count,
                               void *out_palettes, mrw_palette_format out_format, float *model) {
    uint32_t jc = skel->joint_count;
    for (uint32_t base = 0; base < instance_count; ) {
        uint32_t live = instance_count - base; if (live > MRW_LANES) live = MRW_LANES;
        uint32_t i0[MRW_LANES]; float uu[MRW_LANES]; int st[MRW_LANES];
        for (uint32_t l = 0; l < MRW_LANES; ++l) {
            uint32_t idx = base + (l < live ? l : 0);
            st[l] = mrw_clip_frame(base_clip, times[idx], &i0[l], &uu[l]);
        }
        for (uint32_t j = 0; j < jc; ++j) {
            int is_root = (j == 0);
            uint16_t parent = 0;
            if (!is_root) mrw_skeleton_parent(skel, j, &parent);
            float ib[12]; mrw_skeleton_inverse_bind(skel, j, ib);
            for (uint32_t l = 0; l < MRW_LANES; ++l) {
                uint32_t idx = base + (l < live ? l : 0);
                mrw_trs bp, acc;
                sample_joint(base_clip, j, st[l], i0[l], uu[l], &bp);
                float wc = mrw_clampf(weights[idx], 0.0f, 1.0f);
                float mj = mask ? mrw_clampf(mask[j], 0.0f, 1.0f) : 1.0f;
                float we = mrw_clampf(wc * mj, 0.0f, 1.0f);
                /* identical ops to mrw_pose_accumulate ⇒ bit-exact match */
                float nid[4]; mrw_quat_nlerp(MRW_BB_IDENTITY, delta[j].rot, we, nid);
                float r[4]; mrw_quat_mul(bp.rot, nid, r); mrw_quat_normalize(r);
                for (int k = 0; k < 4; ++k) acc.rot[k] = r[k];
                for (int k = 0; k < 3; ++k) acc.trans[k] = bp.trans[k] + we * delta[j].trans[k];
                for (int k = 0; k < 3; ++k) {
                    float target = bp.scale[k] * delta[j].scale[k];
                    acc.scale[k] = bp.scale[k] + we * (target - bp.scale[k]);
                }
                float local12[12]; mrw_trs_to_affine(&acc, local12);
                compose_scatter_scalar(model, ib, local12, is_root, parent,
                                       j, jc, base + l, l, l < live, out_palettes, out_format);
            }
        }
        base += live;
    }
    return MRW_OK;
}

/* ------------------------------------------------------------------ validate + dispatch */

/* Shared alignment/capacity/finite checks; fills *model and *use_f16c on MRW_OK-to-proceed.
 * Returns MRW_OK with *proceed=0 for the instance_count==0 no-op. */
static mrw_result bb_prep(const mrw_dispatch *disp, const mrw_skeleton_view *skel,
                          uint32_t instance_count, mrw_palette_format out_format,
                          const void *out_palettes, size_t out_palettes_capacity,
                          void *scratch, size_t scratch_capacity,
                          float **model, int *use_f16c, int *proceed) {
    *proceed = 0;
    mrw_result drc = mrw_batch_validate_dispatch(disp);
    if (drc) return drc;
    uint32_t elem = bb_elem(out_format);
    if (elem == 0) return MRW_E_RANGE;
    if (instance_count == 0) return MRW_OK;                       /* clean no-op */
    if (!out_palettes || !scratch) return MRW_E_RANGE;

    uint32_t jc = skel->joint_count;
    mrw_mem_req sreq, oreq;
    mrw_result rc = mrw_batch_clip_to_palette_requirements(jc, instance_count, out_format, &sreq, &oreq);
    if (rc) return rc;
    size_t out_align = (out_format == MRW_PALETTE_F16) ? elem : MRW_ALIGN_ARRAY;
    if (((uintptr_t)scratch      & (MRW_ALIGN_SECTION - 1u)) != 0) return MRW_E_ALIGN;
    if (((uintptr_t)out_palettes & (out_align         - 1u)) != 0) return MRW_E_ALIGN;
    if (scratch_capacity      < sreq.size) return MRW_E_CAPACITY;
    if (out_palettes_capacity < oreq.size) return MRW_E_CAPACITY;

    *model = (float *)scratch;
    *use_f16c = (disp->features & MRW_FEAT_F16C) != 0;
    *proceed = 1;
    return MRW_OK;
}

static mrw_result blend_impl(const mrw_dispatch *disp, const mrw_skeleton_view *skel,
                             const mrw_clip_view *clipA, const mrw_clip_view *clipB,
                             const float *timesA, const float *timesB, const float *weights,
                             const float *mask, uint32_t instance_count, mrw_palette_format out_format,
                             void *out_palettes, size_t out_palettes_capacity,
                             void *scratch, size_t scratch_capacity) {
    if (!disp || !skel || !clipA || !clipB) return MRW_E_RANGE;
    uint32_t jc = skel->joint_count;
    if (jc != clipA->joint_count || jc != clipB->joint_count) return MRW_E_INCOMPATIBLE;
    if (!mrw_id_equal(&clipA->skeleton_id, &skel->id) ||
        !mrw_id_equal(&clipB->skeleton_id, &skel->id)) return MRW_E_INCOMPATIBLE;

    float *model = NULL; int use_f16c = 0, proceed = 0;
    mrw_result rc = bb_prep(disp, skel, instance_count, out_format,
                            out_palettes, out_palettes_capacity, scratch, scratch_capacity,
                            &model, &use_f16c, &proceed);
    if (rc || !proceed) return rc;
    if (!timesA || !timesB || !weights) return MRW_E_RANGE;

    for (uint32_t i = 0; i < instance_count; ++i)
        if (!mrw_f32_finite(timesA[i]) || !mrw_f32_finite(timesB[i]) || !mrw_f32_finite(weights[i]))
            return MRW_E_RANGE;
    if (mask) for (uint32_t j = 0; j < jc; ++j) if (!mrw_f32_finite(mask[j])) return MRW_E_RANGE;

    switch (disp->backend) {
        case MRW_BACKEND_SSE2:
            return mrw_blend_kernel_sse2(skel, clipA, clipB, timesA, timesB, weights, mask,
                                         instance_count, out_palettes, out_format, use_f16c, model);
        case MRW_BACKEND_AVX2:
            return (disp->features & MRW_FEAT_FMA)
                 ? mrw_blend_kernel_avx2_fma(skel, clipA, clipB, timesA, timesB, weights, mask,
                                             instance_count, out_palettes, out_format, use_f16c, model)
                 : mrw_blend_kernel_avx2    (skel, clipA, clipB, timesA, timesB, weights, mask,
                                             instance_count, out_palettes, out_format, use_f16c, model);
        default:
            return blend_scalar(skel, clipA, clipB, timesA, timesB, weights, mask,
                                instance_count, out_palettes, out_format, model);
    }
}

static mrw_result accum_impl(const mrw_dispatch *disp, const mrw_skeleton_view *skel,
                             const mrw_clip_view *base_clip, const float *times, const mrw_trs *delta,
                             const float *weights, const float *mask, uint32_t instance_count,
                             mrw_palette_format out_format,
                             void *out_palettes, size_t out_palettes_capacity,
                             void *scratch, size_t scratch_capacity) {
    if (!disp || !skel || !base_clip) return MRW_E_RANGE;
    uint32_t jc = skel->joint_count;
    if (jc != base_clip->joint_count) return MRW_E_INCOMPATIBLE;
    if (!mrw_id_equal(&base_clip->skeleton_id, &skel->id)) return MRW_E_INCOMPATIBLE;

    float *model = NULL; int use_f16c = 0, proceed = 0;
    mrw_result rc = bb_prep(disp, skel, instance_count, out_format,
                            out_palettes, out_palettes_capacity, scratch, scratch_capacity,
                            &model, &use_f16c, &proceed);
    if (rc || !proceed) return rc;
    if (!times || !delta || !weights) return MRW_E_RANGE;

    for (uint32_t i = 0; i < instance_count; ++i)
        if (!mrw_f32_finite(times[i]) || !mrw_f32_finite(weights[i])) return MRW_E_RANGE;
    for (uint32_t j = 0; j < jc; ++j) {
        for (int k = 0; k < 4; ++k) if (!mrw_f32_finite(delta[j].rot[k]))   return MRW_E_RANGE;
        for (int k = 0; k < 3; ++k) if (!mrw_f32_finite(delta[j].trans[k])) return MRW_E_RANGE;
        for (int k = 0; k < 3; ++k) if (!mrw_f32_finite(delta[j].scale[k])) return MRW_E_RANGE;
    }
    if (mask) for (uint32_t j = 0; j < jc; ++j) if (!mrw_f32_finite(mask[j])) return MRW_E_RANGE;

    switch (disp->backend) {
        case MRW_BACKEND_SSE2:
            return mrw_accum_kernel_sse2(skel, base_clip, times, delta, weights, mask,
                                         instance_count, out_palettes, out_format, use_f16c, model);
        case MRW_BACKEND_AVX2:
            return (disp->features & MRW_FEAT_FMA)
                 ? mrw_accum_kernel_avx2_fma(skel, base_clip, times, delta, weights, mask,
                                             instance_count, out_palettes, out_format, use_f16c, model)
                 : mrw_accum_kernel_avx2    (skel, base_clip, times, delta, weights, mask,
                                             instance_count, out_palettes, out_format, use_f16c, model);
        default:
            return accum_scalar(skel, base_clip, times, delta, weights, mask,
                                instance_count, out_palettes, out_format, model);
    }
}

/* ------------------------------------------------------------------ public entry points */

mrw_result mrw_batch_blend_clips_to_palette(
    const mrw_dispatch *disp, const mrw_skeleton_view *skel,
    const mrw_clip_view *clipA, const mrw_clip_view *clipB,
    const float *timesA, const float *timesB, const float *weights, const float *mask,
    uint32_t instance_count, float *out_palettes, size_t out_palettes_capacity,
    void *scratch, size_t scratch_capacity) {
    return blend_impl(disp, skel, clipA, clipB, timesA, timesB, weights, mask, instance_count,
                      MRW_PALETTE_F32, out_palettes, out_palettes_capacity, scratch, scratch_capacity);
}
mrw_result mrw_batch_blend_clips_to_palette_f16(
    const mrw_dispatch *disp, const mrw_skeleton_view *skel,
    const mrw_clip_view *clipA, const mrw_clip_view *clipB,
    const float *timesA, const float *timesB, const float *weights, const float *mask,
    uint32_t instance_count, uint16_t *out_palettes, size_t out_palettes_capacity,
    void *scratch, size_t scratch_capacity) {
    return blend_impl(disp, skel, clipA, clipB, timesA, timesB, weights, mask, instance_count,
                      MRW_PALETTE_F16, out_palettes, out_palettes_capacity, scratch, scratch_capacity);
}
mrw_result mrw_batch_accumulate_to_palette(
    const mrw_dispatch *disp, const mrw_skeleton_view *skel, const mrw_clip_view *base_clip,
    const float *times, const mrw_trs *delta, const float *weights, const float *mask,
    uint32_t instance_count, float *out_palettes, size_t out_palettes_capacity,
    void *scratch, size_t scratch_capacity) {
    return accum_impl(disp, skel, base_clip, times, delta, weights, mask, instance_count,
                      MRW_PALETTE_F32, out_palettes, out_palettes_capacity, scratch, scratch_capacity);
}
mrw_result mrw_batch_accumulate_to_palette_f16(
    const mrw_dispatch *disp, const mrw_skeleton_view *skel, const mrw_clip_view *base_clip,
    const float *times, const mrw_trs *delta, const float *weights, const float *mask,
    uint32_t instance_count, uint16_t *out_palettes, size_t out_palettes_capacity,
    void *scratch, size_t scratch_capacity) {
    return accum_impl(disp, skel, base_clip, times, delta, weights, mask, instance_count,
                      MRW_PALETTE_F16, out_palettes, out_palettes_capacity, scratch, scratch_capacity);
}
