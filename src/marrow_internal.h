/* marrow internal shared declarations - NOT part of the public ABI.
 *
 * Wire-format constants and strict-aliasing-safe little-endian readers. Every
 * multi-byte read goes through memcpy into a native value: we never cast a
 * `float`/`uint32_t` pointer over the caller's byte buffer (a strict-aliasing
 * violation when the storage is a uint8_t array). A constant-size memcpy folds to a
 * single load.
 */
#ifndef MRW_INTERNAL_H
#define MRW_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <string.h> /* memcpy */
#include <math.h>   /* sqrtf, floorf, fmodf - for the shared sampling helpers below */

#include "marrow.h"

/* Layout assumptions the wire readers and struct-memcpy paths depend on:
 * binary32 floats, 8-bit bytes, and padding-free transform/id structs. */
_Static_assert(sizeof(float) == 4, "marrow requires IEEE-754 binary32 float");
_Static_assert(sizeof(mrw_id128) == 16, "mrw_id128 must be 16 bytes");
_Static_assert(sizeof(mrw_trs) == 40, "mrw_trs must match the 40-byte wire TRS");
_Static_assert(sizeof(mrw_xform) == 32, "mrw_xform must be packed quat+trans+scale");

/* ---- wire layout, all little-endian ---- */
enum {
    MRW_MAGIC0 = (uint8_t)'M', MRW_MAGIC1 = (uint8_t)'R',
    MRW_MAGIC2 = (uint8_t)'R', MRW_MAGIC3 = (uint8_t)'W'
};
#define MRW_ENDIAN_TAG 0x04030201u
#define MRW_VERSION    0u

enum {
    MRW_FILE_HEADER_SIZE   = 64,
    MRW_SECTION_ENTRY_SIZE = 32,
    MRW_SECTION_HEADER_SIZE = 128,

    /* per-element strides */
    MRW_PARENT_STRIDE       = 2,   /* u16                       */
    MRW_REST_LOCAL_STRIDE   = 40,  /* q4 + t3 + s3 (f32)        */
    MRW_INVERSE_BIND_STRIDE = 48,  /* 12 f32 (3×4)              */
    MRW_NAME_OFF_STRIDE     = 4,   /* u32                       */
    MRW_CLIP_SAMPLE_STRIDE  = 40,  /* q4 + t3 + s3 (f32)        */
    MRW_ROOT_SAMPLE_STRIDE  = 28,  /* q4 + t3 (f32)             */
    MRW_TEXEL_STRIDE        = 8,   /* RGBA16F                   */
    MRW_BAKED_CLIP_STRIDE   = 48,  /* clip-table entry          */

    MRW_ALIGN_SECTION = 64,
    MRW_ALIGN_ARRAY   = 16
};

/* ---- little-endian readers (host is little-endian) ---- */
static inline uint16_t mrw_rd_u16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t mrw_rd_u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t mrw_rd_u64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline float    mrw_rd_f32(const uint8_t *p) { float    v; memcpy(&v, p, 4); return v; }

static inline mrw_id128 mrw_rd_id128(const uint8_t *p) {
    mrw_id128 id; id.lo = mrw_rd_u64(p); id.hi = mrw_rd_u64(p + 8); return id;
}

/* ---- value predicates (no <math.h> dependency) ---- */
static inline int mrw_f32_finite(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    return ((b >> 23) & 0xFFu) != 0xFFu; /* exponent all-ones ⇒ inf/nan */
}
/* |‖q‖ − 1| ≤ tol, checked on the squared norm to avoid sqrt. */
static inline int mrw_quat_near_unit(float x, float y, float z, float w, float tol) {
    float s  = x * x + y * y + z * z + w * w;
    float lo = (1.0f - tol) * (1.0f - tol);
    float hi = (1.0f + tol) * (1.0f + tol);
    return s >= lo && s <= hi;
}

/* ---- shared clip-sampling helpers ----
 * One source of truth for the clip-pose sampling math, used by BOTH the single-instance
 * pose path (marrow_pose.c) and the homogeneous batch path (marrow_batch.c) so the two are
 * bit-identical. `static inline` ⇒ each TU compiles its own internal-linkage copy: zero
 * added exported ABI symbols. */

static inline float mrw_clampf(float t, float lo, float hi) {
    return t < lo ? lo : (t > hi ? hi : t);
}

/* mod_pos(t,D) = ((t mod D)+D) mod D, kept strictly in [0,D): the +D can round to exactly D
 * in float, so the trailing subtract re-applies the outer modulo. */
static inline float mrw_mod_pos(float t, float d) {
    float m = fmodf(t, d);
    if (m < 0.0f) m += d;
    if (m >= d) m -= d;
    return m;
}

/* hemisphere-corrected nlerp of two quaternions */
static inline void mrw_quat_nlerp(const float a[4], const float b[4], float u, float out[4]) {
    float dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    float s = (dot < 0.0f) ? -1.0f : 1.0f;
    float q[4];
    for (int k = 0; k < 4; ++k) q[k] = a[k] + u * (s * b[k] - a[k]);
    float n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    float inv = (n2 > 0.0f) ? 1.0f / sqrtf(n2) : 0.0f;
    for (int k = 0; k < 4; ++k) out[k] = q[k] * inv;
}

/* interpolate a full TRS (clip pose): nlerp quat, lerp translation+scale */
static inline void mrw_interp_trs(const mrw_trs *a, const mrw_trs *b, float u, mrw_trs *o) {
    mrw_quat_nlerp(a->rot, b->rot, u, o->rot);
    for (int k = 0; k < 3; ++k) {
        o->trans[k] = a->trans[k] + u * (b->trans[k] - a->trans[k]);
        o->scale[k] = a->scale[k] + u * (b->scale[k] - a->scale[k]);
    }
}

/* frame indexing for a CLIP pose track. Returns 1 if the track is static
 * (count==1 or duration==0 ⇒ use sample 0); otherwise sets *i0 and *u.
 * *i0/*u are defined on every path (0 on the static returns): callers store them into
 * per-lane arrays and forward them unconditionally, reading them only when 0 is returned,
 * so the static path must not leave the out-params indeterminate. */
static inline int mrw_clip_frame(const mrw_clip_view *clip, float t, uint32_t *i0, float *u) {
    *i0 = 0; *u = 0.0f;
    uint32_t n = clip->sample_count;
    if (n == 1) return 1;
    float duration = (float)(n - 1) / clip->fps;
    if (duration == 0.0f) return 1;
    int looping = (clip->flags & MRW_CLIP_LOOPING) != 0;
    float t_local = looping ? mrw_mod_pos(t, duration) : mrw_clampf(t, 0.0f, duration);
    float fpos = (t_local / duration) * (float)(n - 1);
    uint32_t i = (uint32_t)floorf(fpos);
    if (i > n - 2) i = n - 2;
    *i0 = i;
    *u = fpos - (float)i;
    return 0;
}

/* ---- batch SoA scratch layout ----
 * Per-joint palette/affine entry: 12 binary32 (three vec4 rows). The model scratch is
 * across-instance SoA, lane (instance) innermost, so a `wide` load grabs MRW_LANES instances of
 * one component. Shared by the scalar kernel (marrow_batch.c) and the SIMD kernels. */
#define MRW_PALETTE_FLOATS 12u
#define MRW_MODEL_AT(model, j, c, l) ((model)[((size_t)(j) * MRW_PALETTE_FLOATS + (c)) * MRW_LANES + (l)])

/* SIMD batch kernels (defined in the ISA-flagged TUs marrow_batch_{sse2,avx2,avx2_fma}.c;
 * called by the dispatcher in marrow_batch.c AFTER it has validated all inputs). `out_palettes` is
 * float* (MRW_PALETTE_F32) or uint16_t* (MRW_PALETTE_F16); `use_f16c` lets the AVX2 store narrow with
 * vcvtps2ph (else the parity-equal scalar mrw_f32_to_f16 fallback; ignored by SSE2/scalar, which stay
 * pure). `model` is the validated SoA scratch (always f32). Not part of the public ABI. */
mrw_result mrw_batch_kernel_sse2    (const mrw_skeleton_view *skel, const mrw_clip_view *clip,
                                     const float *times, uint32_t instance_count,
                                     void *out_palettes, mrw_palette_format out_format,
                                     int use_f16c, float *model);
mrw_result mrw_batch_kernel_avx2    (const mrw_skeleton_view *skel, const mrw_clip_view *clip,
                                     const float *times, uint32_t instance_count,
                                     void *out_palettes, mrw_palette_format out_format,
                                     int use_f16c, float *model);
mrw_result mrw_batch_kernel_avx2_fma(const mrw_skeleton_view *skel, const mrw_clip_view *clip,
                                     const float *times, uint32_t instance_count,
                                     void *out_palettes, mrw_palette_format out_format,
                                     int use_f16c, float *model);

/* pose-combine SIMD kernels (defined in the ISA-flagged TUs marrow_blend_{sse2,avx2,avx2_fma}.c
 * via the shared marrow_blend_simd.h; called by the dispatcher in marrow_blend_batch.c AFTER it has
 * validated all inputs). Same SoA scratch / out_palettes / use_f16c contract as the clip kernels.
 * blend: per instance blend(sample(clipA),sample(clipB),weight) with a shared per-joint mask.
 * accum: per instance accumulate(sample(base_clip), shared delta, weight, mask). Not public ABI. */
/* Shared dispatch validation (defined in marrow_batch.c): `features` MUST be the canonical
 * set for `backend`, else MRW_E_UNSUPPORTED. Reused by the pose-combine dispatcher. */
mrw_result mrw_batch_validate_dispatch(const mrw_dispatch *d);

#define MRW_DECL_BLEND_KERNEL(name) \
    mrw_result name(const mrw_skeleton_view *skel, \
                    const mrw_clip_view *clipA, const mrw_clip_view *clipB, \
                    const float *timesA, const float *timesB, const float *weights, const float *mask, \
                    uint32_t instance_count, void *out_palettes, mrw_palette_format out_format, \
                    int use_f16c, float *model)
#define MRW_DECL_ACCUM_KERNEL(name) \
    mrw_result name(const mrw_skeleton_view *skel, const mrw_clip_view *base_clip, \
                    const float *times, const mrw_trs *delta, const float *weights, const float *mask, \
                    uint32_t instance_count, void *out_palettes, mrw_palette_format out_format, \
                    int use_f16c, float *model)
MRW_DECL_BLEND_KERNEL(mrw_blend_kernel_sse2);
MRW_DECL_BLEND_KERNEL(mrw_blend_kernel_avx2);
MRW_DECL_BLEND_KERNEL(mrw_blend_kernel_avx2_fma);
MRW_DECL_ACCUM_KERNEL(mrw_accum_kernel_sse2);
MRW_DECL_ACCUM_KERNEL(mrw_accum_kernel_avx2);
MRW_DECL_ACCUM_KERNEL(mrw_accum_kernel_avx2_fma);

#endif /* MRW_INTERNAL_H */
