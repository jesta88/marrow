/* marrow homogeneous batch - animate N instances sharing one skeleton + clip, each at its own
 * time, into the per-instance joint-contiguous AoS 3×4 palette. Scalar reference over the
 * internal across-instance SoA scratch: lane (instance) is the innermost axis, so the SSE2/AVX2
 * kernels load 8 instances of one component at once. The result is bit-for-bit equal to looping
 * mrw_clip_to_palette over the instances. */
#include "marrow_internal.h"

/* ------------------------------------------------------------------ overflow-safe sizing */

/* p = a*b with uint64 overflow detection (counts are u32, but their products are not). */
static int u64_mul(uint64_t a, uint64_t b, uint64_t *p) {
    if (b != 0 && a > UINT64_MAX / b) return 0;
    *p = a * b;
    return 1;
}
static int u64_round_up_64(uint64_t v, uint64_t *r) {
    if (v > UINT64_MAX - 63u) return 0;
    *r = (v + 63u) & ~(uint64_t)63u;
    return 1;
}
static int u64_to_size(uint64_t v, size_t *out) {
    if (v > (uint64_t)SIZE_MAX) return 0;
    *out = (size_t)v;
    return 1;
}

/* Required scratch and output byte sizes; MRW_E_OVERFLOW on any size-math overflow. `out_elem` is the
 * output palette element size (4 = F32, 2 = F16) - only the OUTPUT scales with it; the SoA scratch is
 * always f32 (the hierarchy composes in f32; only the final AoS store narrows).
 * scratch_bytes / out_bytes may be NULL if not wanted. */
static mrw_result batch_sizes(uint32_t joint_count, uint32_t instance_count, uint32_t out_elem,
                              size_t *scratch_bytes, size_t *out_bytes) {
    /* scratch: joint_count × 12 × MRW_LANES floats, cache-line rounded. */
    if (scratch_bytes) {
        uint64_t v;
        if (!u64_mul(joint_count, MRW_PALETTE_FLOATS, &v)) return MRW_E_OVERFLOW;
        if (!u64_mul(v, MRW_LANES, &v))                    return MRW_E_OVERFLOW;
        if (!u64_mul(v, sizeof(float), &v))                return MRW_E_OVERFLOW;
        if (!u64_round_up_64(v, &v))                       return MRW_E_OVERFLOW;
        if (!u64_to_size(v, scratch_bytes))                return MRW_E_OVERFLOW;
    }
    /* output: instance_count × joint_count × 12 components × out_elem bytes, AoS. */
    if (out_bytes) {
        uint64_t v;
        if (!u64_mul(instance_count, joint_count, &v))     return MRW_E_OVERFLOW;
        if (!u64_mul(v, MRW_PALETTE_FLOATS, &v))           return MRW_E_OVERFLOW;
        if (!u64_mul(v, out_elem, &v))                     return MRW_E_OVERFLOW;
        if (!u64_to_size(v, out_bytes))                    return MRW_E_OVERFLOW;
    }
    return MRW_OK;
}

/* output element size (bytes) for a palette format; 0 ⇒ out-of-range format. */
static uint32_t palette_elem(mrw_palette_format f) {
    return f == MRW_PALETTE_F32 ? 4u : f == MRW_PALETTE_F16 ? 2u : 0u;
}

mrw_result mrw_batch_clip_to_palette_requirements(
    uint32_t joint_count, uint32_t instance_count, mrw_palette_format out_format,
    mrw_mem_req *out_scratch, mrw_mem_req *out_palettes) {
    if (!out_scratch && !out_palettes) return MRW_E_RANGE;
    uint32_t elem = palette_elem(out_format);
    if (elem == 0) return MRW_E_RANGE;
    size_t sbytes = 0, obytes = 0;
    mrw_result rc = batch_sizes(joint_count, instance_count, elem,
                                out_scratch ? &sbytes : NULL,
                                out_palettes ? &obytes : NULL);
    if (rc) return rc;
    if (out_scratch)  { out_scratch->size  = sbytes; out_scratch->align  = MRW_ALIGN_SECTION; } /* 64 */
    if (out_palettes) { out_palettes->size = obytes; out_palettes->align = MRW_ALIGN_ARRAY;   } /* 16 */
    return MRW_OK;
}

/* ------------------------------------------------------------------ dispatch validation */

/* `features` must be EXACTLY the canonical set a constructor produces for `backend`.
 * This is the honest reading of "self-consistent": reserved bits, an AVX2 dispatch missing its
 * implied SSE2/AVX/YMM bits (which would still reach the FMA-routed kernel), or feature bits that
 * contradict the backend all ⇒ MRW_E_UNSUPPORTED. It does NOT (cannot) check the host - a valid
 * dispatch is only constructible via mrw_dispatch_detect/_sse2/_avx2, which do; fabricating one
 * that lies about the host is UB by contract - but no malformed/contradictory value gets through. */
mrw_result mrw_batch_validate_dispatch(const mrw_dispatch *d) {
    const uint32_t avx2_base = MRW_FEAT_SSE2 | MRW_FEAT_AVX | MRW_FEAT_AVX2 | MRW_FEAT_OSXSAVE_YMM;
    switch (d->backend) {
        case MRW_BACKEND_SCALAR: return (d->features == 0) ? MRW_OK : MRW_E_UNSUPPORTED;
        case MRW_BACKEND_SSE2:   return (d->features == MRW_FEAT_SSE2) ? MRW_OK : MRW_E_UNSUPPORTED;
        /* FMA and F16C are independent optional refinements of the AVX2 backend (FMA picks the fused
         * kernel; F16C accelerates only the f16 store). All four combos are canonical. */
        case MRW_BACKEND_AVX2:   return (d->features == avx2_base ||
                                         d->features == (avx2_base | MRW_FEAT_FMA) ||
                                         d->features == (avx2_base | MRW_FEAT_F16C) ||
                                         d->features == (avx2_base | MRW_FEAT_FMA | MRW_FEAT_F16C))
                                        ? MRW_OK : MRW_E_UNSUPPORTED;
        default: return MRW_E_UNSUPPORTED;
    }
}

/* ------------------------------------------------------------------ scalar SoA kernel (reference) */

/* Same scalar ops, same order, per lane → bit-identical to looping mrw_clip_to_palette; the SIMD
 * kernels are checked against it. Inputs are pre-validated by mrw_batch_clip_to_palette;
 * `model` is the validated SoA scratch. */
static mrw_result batch_scalar(const mrw_skeleton_view *skel, const mrw_clip_view *clip,
                               const float *times, uint32_t instance_count,
                               void *out_palettes, mrw_palette_format out_format, float *model) {
    uint32_t jc = skel->joint_count;
    for (uint32_t base = 0; base < instance_count; ) {
        uint32_t live = instance_count - base;
        if (live > MRW_LANES) live = MRW_LANES;

        /* Per-lane frame state. Padded lanes [live, MRW_LANES) take a SAFE in-range time
         * (times[base]) - never an OOB read of `times` - and are computed but never scattered. */
        uint32_t i0[MRW_LANES]; float uu[MRW_LANES]; int st[MRW_LANES];
        for (uint32_t l = 0; l < MRW_LANES; ++l) {
            float t = times[base + (l < live ? l : 0)];
            st[l] = mrw_clip_frame(clip, t, &i0[l], &uu[l]);
        }

        for (uint32_t j = 0; j < jc; ++j) {
            int is_root = (j == 0);
            uint16_t parent = 0;
            if (!is_root) mrw_skeleton_parent(skel, j, &parent); /* parent < j (validated) */

            /* Compose model(j) per lane: model(j) = root ? local : model(parent) ∘ local. */
            for (uint32_t l = 0; l < MRW_LANES; ++l) {
                mrw_trs lj;
                if (st[l]) {
                    mrw_clip_sample(clip, j, 0, &lj);
                } else {
                    mrw_trs a, b;
                    mrw_clip_sample(clip, j, i0[l], &a);
                    mrw_clip_sample(clip, j, i0[l] + 1, &b);
                    mrw_interp_trs(&a, &b, uu[l], &lj);
                }
                float local12[MRW_PALETTE_FLOATS], mj[MRW_PALETTE_FLOATS];
                mrw_trs_to_affine(&lj, local12);
                if (is_root) {
                    memcpy(mj, local12, sizeof mj);
                } else {
                    float pm[MRW_PALETTE_FLOATS];
                    for (uint32_t c = 0; c < MRW_PALETTE_FLOATS; ++c) pm[c] = MRW_MODEL_AT(model, parent, c, l);
                    mrw_affine_mul(pm, local12, mj);
                }
                for (uint32_t c = 0; c < MRW_PALETTE_FLOATS; ++c) MRW_MODEL_AT(model, j, c, l) = mj[c];
            }

            /* Scatter M_j = model(j) ∘ inverse_bind(j) to the AoS palette - LIVE lanes
             * only, so padded lanes can never touch caller output. */
            float ib[MRW_PALETTE_FLOATS];
            mrw_skeleton_inverse_bind(skel, j, ib);
            for (uint32_t l = 0; l < live; ++l) {
                float mj[MRW_PALETTE_FLOATS], M[MRW_PALETTE_FLOATS];
                for (uint32_t c = 0; c < MRW_PALETTE_FLOATS; ++c) mj[c] = MRW_MODEL_AT(model, j, c, l);
                mrw_affine_mul(mj, ib, M);
                size_t off = ((size_t)(base + l) * jc + j) * MRW_PALETTE_FLOATS;
                if (out_format == MRW_PALETTE_F16) {
                    uint16_t *dst = (uint16_t *)out_palettes + off;
                    for (uint32_t c = 0; c < MRW_PALETTE_FLOATS; ++c) dst[c] = mrw_f32_to_f16(M[c]);
                } else {
                    memcpy((float *)out_palettes + off, M, sizeof M);
                }
            }
        }
        base += live; /* overflow-safe: base only ever reaches instance_count (no +8 wrap) */
    }
    return MRW_OK;
}

/* ------------------------------------------------------------------ public entry: validate + dispatch */

/* Shared validate + dispatch for both output formats; out_palettes is float* (F32) or uint16_t* (F16).
 * All validation, scratch sizing, and backend selection are format-independent - only the output
 * element size (and the final store narrowing inside the kernels) differs. */
static mrw_result batch_impl(
    const mrw_dispatch *disp,
    const mrw_skeleton_view *skel, const mrw_clip_view *clip,
    const float *times, uint32_t instance_count, mrw_palette_format out_format,
    void *out_palettes, size_t out_palettes_capacity,
    void *scratch,      size_t scratch_capacity) {

    if (!disp || !skel || !clip) return MRW_E_RANGE;
    if (skel->joint_count != clip->joint_count) return MRW_E_INCOMPATIBLE;
    if (!mrw_id_equal(&clip->skeleton_id, &skel->id)) return MRW_E_INCOMPATIBLE;

    /* Validate the caller-owned dispatch before any ISA-specific code; a forced backend
     * without the matching feature bits → MRW_E_UNSUPPORTED (no silent downgrade). */
    mrw_result drc = mrw_batch_validate_dispatch(disp);
    if (drc) return drc;

    uint32_t elem = palette_elem(out_format);
    if (elem == 0) return MRW_E_RANGE;

    /* Empty batch is a clean no-op; the data pointers may legitimately be NULL/zero-capacity. */
    if (instance_count == 0) return MRW_OK;

    if (!times || !out_palettes || !scratch) return MRW_E_RANGE;

    uint32_t jc = skel->joint_count;
    size_t need_scratch = 0, need_out = 0;
    mrw_result rc = batch_sizes(jc, instance_count, elem, &need_scratch, &need_out);
    if (rc) return rc;

    /* Alignment is well-defined via uintptr_t (as mrw_blob_open checks the blob base). The
     * requirements query recommends a 16-byte (cache/AVX-friendly) base for either format. f32 keeps
     * that as a hard precondition; the f16 scatter is FULLY unaligned (storeu/storel, scalar
     * per-component), so a 24-B-stride sub-range pointer (only `elem`-aligned on odd joint counts) is
     * valid. Accepting element alignment for f16 lets a caller write disjoint sub-ranges of one larger
     * palette in place - the crowd batch path - without a staging copy. */
    size_t out_align = (out_format == MRW_PALETTE_F16) ? elem : MRW_ALIGN_ARRAY;
    if (((uintptr_t)scratch      & (MRW_ALIGN_SECTION - 1u)) != 0) return MRW_E_ALIGN; /* 64 */
    if (((uintptr_t)out_palettes & (out_align         - 1u)) != 0) return MRW_E_ALIGN;
    if (scratch_capacity      < need_scratch) return MRW_E_CAPACITY;
    if (out_palettes_capacity < need_out)     return MRW_E_CAPACITY;

    /* All-or-nothing: reject any non-finite time before writing output. */
    for (uint32_t i = 0; i < instance_count; ++i)
        if (!mrw_f32_finite(times[i])) return MRW_E_RANGE;

    float *model = (float *)scratch; /* across-instance SoA, joint_count × 12 × MRW_LANES (always f32) */
    int use_f16c = (disp->features & MRW_FEAT_F16C) != 0; /* only the AVX2 f16 store consults this */

    switch (disp->backend) {
        case MRW_BACKEND_SSE2:
            return mrw_batch_kernel_sse2(skel, clip, times, instance_count,
                                         out_palettes, out_format, use_f16c, model);
        case MRW_BACKEND_AVX2:
            return (disp->features & MRW_FEAT_FMA)
                 ? mrw_batch_kernel_avx2_fma(skel, clip, times, instance_count,
                                             out_palettes, out_format, use_f16c, model)
                 : mrw_batch_kernel_avx2    (skel, clip, times, instance_count,
                                             out_palettes, out_format, use_f16c, model);
        default: /* SCALAR - validate_dispatch already rejected anything else */
            return batch_scalar(skel, clip, times, instance_count, out_palettes, out_format, model);
    }
}

mrw_result mrw_batch_clip_to_palette(
    const mrw_dispatch *disp,
    const mrw_skeleton_view *skel, const mrw_clip_view *clip,
    const float *times, uint32_t instance_count,
    float *out_palettes, size_t out_palettes_capacity,
    void  *scratch,      size_t scratch_capacity) {
    return batch_impl(disp, skel, clip, times, instance_count, MRW_PALETTE_F32,
                      out_palettes, out_palettes_capacity, scratch, scratch_capacity);
}

mrw_result mrw_batch_clip_to_palette_f16(
    const mrw_dispatch *disp,
    const mrw_skeleton_view *skel, const mrw_clip_view *clip,
    const float *times, uint32_t instance_count,
    uint16_t *out_palettes, size_t out_palettes_capacity,
    void     *scratch,      size_t scratch_capacity) {
    return batch_impl(disp, skel, clip, times, instance_count, MRW_PALETTE_F16,
                      out_palettes, out_palettes_capacity, scratch, scratch_capacity);
}
