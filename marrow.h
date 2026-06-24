/*
 * marrow
 *
 * Copyright (C) 2026, by Jérémie St-Amand (jeremie.stamand@gmail.com)
 * Report bugs and download new versions at https://github.com/jesta88/marrow
 *
 * This library is distributed under the MIT License. See notice at the end of this file.
 *
 * Conventions: right-handed, +Y up; meters/seconds/radians; column vectors
 * (p' = M·[p,1]); unit quaternions in component order (x,y,z,w), Hamilton
 * product; local compose order local = T·R·S; little-endian, IEEE-754.
 *
 * ABI: every output goes through out-params plus a mrw_result return code;
 * functions never return a nontrivial struct by value; counts are unsigned
 * and fixed-width; no compiler SIMD types appear in public structs; the
 * header compiles as both C11 and C++.
 */
#ifndef MRW_H
#define MRW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MRW_VERSION_MAJOR 0
#define MRW_VERSION_MINOR 1
#define MRW_VERSION_PATCH 0
#define MRW_VERSION_STRING "0.1.0"

/* `restrict` only via a C++-safe macro (C++ has no `restrict` keyword). */
#if defined(__cplusplus)
#  define MRW_RESTRICT
#elif defined(_MSC_VER)
#  define MRW_RESTRICT __restrict
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#  define MRW_RESTRICT restrict
#else
#  define MRW_RESTRICT
#endif

/* Batch lane width - identical for every backend. A logical count N pads up to
 * round_up(N, MRW_LANES), so the buffer-size queries are dispatch-independent. */
#define MRW_LANES 8u

/* ------------------------------------------------------------------ result codes */

typedef uint32_t mrw_result;
enum {
    MRW_OK = 0,
    MRW_E_FORMAT,       /* bad magic/version/endianness/flag/structural invariant */
    MRW_E_RANGE,        /* out-of-range index/offset/count, or non-finite time     */
    MRW_E_ALIGN,        /* blob base or an internal offset is misaligned           */
    MRW_E_OVERFLOW,     /* size/offset arithmetic would overflow                   */
    MRW_E_CAPACITY,     /* caller buffer too small                                 */
    MRW_E_UNSUPPORTED,  /* known-but-not-enabled feature (codec/encoding/section)  */
    MRW_E_INCOMPATIBLE  /* skeleton/clip/baked identity mismatch, or section absent */
};

/* ------------------------------------------------------------------ dispatch */

typedef uint32_t mrw_backend;
enum {
    MRW_BACKEND_SCALAR = 0,
    MRW_BACKEND_SSE2   = 1,
    MRW_BACKEND_AVX2   = 2
};

enum { /* mrw_dispatch.features bits; all other bits reserved, MUST be 0 */
    MRW_FEAT_SSE2        = 1u << 0,
    MRW_FEAT_AVX         = 1u << 1,
    MRW_FEAT_AVX2        = 1u << 2,
    MRW_FEAT_FMA         = 1u << 3,
    MRW_FEAT_OSXSAVE_YMM = 1u << 4,
    MRW_FEAT_F16C        = 1u << 5  /* CPUID.1:ECX[29] - vcvtps2ph for the f16 palette store */
};

/* POD value type - no SIMD types, no lazy globals; caller-owned and immutable. */
typedef struct mrw_dispatch { uint32_t backend; uint32_t features; } mrw_dispatch;

/* Detection checks CPUID AVX/AVX2, OSXSAVE+enabled YMM state, and FMA and F16C independently.
 * _detect picks the best *implemented* backend the host supports (AVX2(+FMA)(+F16C) ▸ SSE2 ▸
 * scalar) and sets `features` to that backend's canonical bits. F16C is an optional refinement of
 * the AVX2 backend (it only speeds the f16 palette store; the scalar mrw_f32_to_f16 fallback is
 * parity-equal), so an AVX2 dispatch is valid with or without it. The forced constructors validate
 * against the host: an unsupported ISA yields MRW_E_UNSUPPORTED and a zeroed *out. _scalar is
 * infallible. */
mrw_result mrw_dispatch_detect(mrw_dispatch *out);
mrw_result mrw_dispatch_scalar(mrw_dispatch *out);
mrw_result mrw_dispatch_sse2  (mrw_dispatch *out);
mrw_result mrw_dispatch_avx2  (mrw_dispatch *out);

/* ------------------------------------------------------------------ value types */

/* 128-bit content fingerprint; opaque, compared only for equality. */
typedef struct mrw_id128 { uint64_t lo, hi; } mrw_id128;

/* Rigid+uniform-scale transform: quat(xyzw) + translation + one scale.
 * Used by the baked encoding and root-motion deltas. */
typedef struct mrw_xform { float rot[4]; float trans[3]; float scale; } mrw_xform;

/* General local transform: quat(xyzw) + translation + NON-uniform scale.
 * Exactly the 40-byte wire layout of a skeleton rest_local / clip sample. */
typedef struct mrw_trs { float rot[4]; float trans[3]; float scale[3]; } mrw_trs;

typedef struct mrw_mem_req { size_t size; size_t align; } mrw_mem_req;

int mrw_id_equal(const mrw_id128 *a, const mrw_id128 *b); /* 1 if equal */

/* ------------------------------------------------------------------ section views */

/* `base` points to the SECTION start (blob + section.offset); all *_off are
 * SECTION-RELATIVE u32 byte offsets to already-range-checked, >=16-aligned arrays.
 * Views BORROW the blob: the caller keeps the immutable blob alive. */
typedef struct mrw_skeleton_view {
    const uint8_t *base; uint32_t joint_count, name_blob_size; mrw_id128 id;
    uint32_t parent_off, rest_local_off, inverse_bind_off, name_off_off, name_blob_off;
} mrw_skeleton_view;

typedef struct mrw_clip_view {
    const uint8_t *base; uint32_t joint_count, codec, sample_count, flags; float fps;
    mrw_id128 id, skeleton_id; uint32_t samples_off, root_track_off; /* root_track_off==0 ⇒ none */
} mrw_clip_view;

typedef struct mrw_baked_view {
    const uint8_t *base; uint32_t encoding, bone_count, texels_per_bone, frame_stride_texels,
    total_frames, clip_count, flags; mrw_id128 skeleton_id;
    uint32_t clip_table_off, texels_off, texel_count;
} mrw_baked_view;

/* Decoded BAKED clip-table entry. */
typedef struct mrw_baked_clip {
    mrw_id128 clip_id; uint32_t first_frame, frame_count; float source_duration_s; uint32_t flags;
} mrw_baked_clip;

/* ------------------------------------------------------------------ section / flag ids */

enum { /* section table `type` */
    MRW_SECTION_SKELETON = 1,
    MRW_SECTION_CLIP     = 2,
    MRW_SECTION_BAKED    = 3
};
enum { MRW_SECTION_FLAG_OPTIONAL = 1u << 0 }; /* a section of unknown type may be skipped */

enum { /* CLIP header flags */
    MRW_CLIP_LOOPING         = 1u << 0,
    MRW_CLIP_HAS_ROOT_MOTION = 1u << 1
};
enum { MRW_BAKED_CLIP_LOOPING = 1u << 0 }; /* baked clip_table entry flag */

/* ------------------------------------------------------------------ blob loader */

/* Validated handle over an immutable .mrw blob. POD, caller-allocated, borrows the
 * blob. Populated by mrw_blob_open, which performs the entire validation pass;
 * after it returns MRW_OK every accessor below is a pure, checked locator. */
typedef struct mrw_blob {
    const uint8_t *base;              /* blob base (>=64-aligned)          */
    uint64_t       size;              /* == provided size                  */
    const uint8_t *section_table;     /* base + section_table_off, or NULL */
    uint32_t       section_count;
    uint32_t       section_table_off; /* avoids base-pointer subtraction   */
} mrw_blob;

/* Validate, then view. `data` MUST be >=64-byte aligned. Does the full deterministic
 * validation pass (header, section table, every section header + arrays, cross-refs). */
mrw_result mrw_blob_open(const void *data, uint64_t size, mrw_blob *out);

/* Section-table accessors (indices in [0, section_count)). */
mrw_result mrw_blob_section_type(const mrw_blob *b, uint32_t index, uint32_t *out_type);

/* Fill a typed view from section `index`; MRW_E_INCOMPATIBLE if the section there
 * is a different type, MRW_E_RANGE if index is out of range. */
mrw_result mrw_skeleton_view_at(const mrw_blob *b, uint32_t index, mrw_skeleton_view *out);
mrw_result mrw_clip_view_at    (const mrw_blob *b, uint32_t index, mrw_clip_view *out);
mrw_result mrw_baked_view_at   (const mrw_blob *b, uint32_t index, mrw_baked_view *out);

/* Convenience lookups (the blob has at most one SKELETON). */
mrw_result mrw_blob_skeleton(const mrw_blob *b, mrw_skeleton_view *out);
mrw_result mrw_blob_skeleton_by_id(const mrw_blob *b, const mrw_id128 *id, mrw_skeleton_view *out);
mrw_result mrw_blob_clip_by_id(const mrw_blob *b, const mrw_id128 *id, mrw_clip_view *out);

/* Typed array readers over a validated view (strict-aliasing-safe; memcpy under the
 * hood). Index/range are not re-checked beyond what open guaranteed. */
mrw_result mrw_skeleton_parent     (const mrw_skeleton_view *v, uint32_t joint, uint16_t *out);
mrw_result mrw_skeleton_rest_local  (const mrw_skeleton_view *v, uint32_t joint, mrw_trs *out);
mrw_result mrw_skeleton_inverse_bind(const mrw_skeleton_view *v, uint32_t joint, float out_affine12[12]);
mrw_result mrw_skeleton_joint_name  (const mrw_skeleton_view *v, uint32_t joint, const char **out_name);
mrw_result mrw_clip_sample          (const mrw_clip_view *v, uint32_t joint, uint32_t sample, mrw_trs *out);
mrw_result mrw_clip_root_sample      (const mrw_clip_view *v, uint32_t sample, mrw_xform *out); /* scale=1 */
mrw_result mrw_baked_clip_entry      (const mrw_baked_view *v, uint32_t clip_index, mrw_baked_clip *out);

/* ------------------------------------------------------------------ identities */

/* FNV-1a-128 over the layout-independent canonical stream. The raw forms take the
 * file arrays directly (id fields are excluded from the stream, so callers building
 * a blob compute the id, then stamp it in). View wrappers recompute from viewed bytes. */
mrw_result mrw_skeleton_compute_id(
    uint32_t joint_count, uint32_t name_blob_size,
    const uint16_t *parent,        /* joint_count                 */
    const float    *rest_local,    /* joint_count * 10 (q,t,s3)   */
    const float    *inverse_bind,  /* joint_count * 12            */
    const uint32_t *name_off,      /* joint_count                 */
    const uint8_t  *name_blob,     /* name_blob_size              */
    mrw_id128 *out);

mrw_result mrw_clip_compute_id(
    uint32_t joint_count, float fps, uint32_t sample_count, uint32_t flags,
    const float *samples,    /* joint_count * sample_count * 10 (joint-major)     */
    const float *root_track, /* sample_count * 7, or NULL if !HAS_ROOT_MOTION     */
    mrw_id128 *out);

mrw_result mrw_skeleton_view_id(const mrw_skeleton_view *v, mrw_id128 *out);
mrw_result mrw_clip_view_id    (const mrw_clip_view *v, mrw_id128 *out);

/* ------------------------------------------------------------------ math core */

/* These are pure, total, infallible primitives - no failure mode and no nontrivial
 * struct returned by value - so, unlike the rest of the API, they do not return
 * mrw_result: outputs go through out-params; the lone scalar return (half decode)
 * is a trivially-copyable value. */
void  mrw_quat_to_mat3   (const float q[4], float out9[9]);            /* unit quat → 3×3 rotation */
void  mrw_trs_to_affine  (const mrw_trs *trs, float out_affine12[12]); /* TRS → 3×4 affine (non-uniform scale) */
void  mrw_xform_to_affine(const mrw_xform *x, float out_affine12[12]); /* xform → 3×4 affine (uniform scale)   */
void  mrw_affine_mul     (const float a12[12], const float b12[12], float out12[12]); /* compose: A∘B (apply B then A) */
void  mrw_cofactor3      (const float a9[9], float out9[9]);           /* 3×3 cofactor matrix (via minors) */
void  mrw_affine_cofactor(const float a12[12], float out9[9]);        /* cofactor of the 3×3 part  */
float    mrw_half_to_float(uint16_t h);                                /* IEEE-754 binary16→f32 */
uint16_t mrw_f32_to_f16   (float f);                                   /* IEEE-754 f32→binary16 (round to nearest even) */

/* ------------------------------------------------------------------ pose pipeline */

/* All four write joint-indexed output. `joint_capacity` is how many joint entries the
 * caller's output (and scratch) buffer holds; the call returns MRW_E_CAPACITY rather
 * than overrun if joint_count exceeds it. Output size is joint_count entries:
 * mrw_trs for sample_local, 12 floats (3×4) per joint otherwise. */

/* Sample a dense clip's local poses at time `t`, interpolating between frames. */
mrw_result mrw_clip_sample_local(const mrw_clip_view *clip, float t,
                                       mrw_trs *out_locals, uint32_t joint_capacity);

/* Hierarchy compose: model(j) = root ? local(j) : model(parent)∘local(j). */
mrw_result mrw_local_to_model(const mrw_skeleton_view *skel, const mrw_trs *locals,
                                    float *out_model, uint32_t joint_capacity);

/* Apply inverse-bind: M_j = model(j) ∘ inverse_bind(j) - the skinning palette. */
mrw_result mrw_model_to_palette(const mrw_skeleton_view *skel, const float *model,
                                      float *out_palette, uint32_t joint_capacity);

/* Fused convenience: sample → local→model → skinning palette. `scratch_model` and
 * `out_palette` are caller buffers of joint_capacity*12 floats (>=16-aligned). */
mrw_result mrw_clip_to_palette(const mrw_skeleton_view *skel, const mrw_clip_view *clip,
                                     float t, float *scratch_model, float *out_palette,
                                     uint32_t joint_capacity);

/* Decode + interpolate one baked bone into a 3×4 affine. `t` is clip-local time;
 * clip_index selects the baked clip_table entry. */
mrw_result mrw_baked_sample_bone(const mrw_baked_view *baked, uint32_t clip_index,
                                       uint32_t bone, float t, float out_affine12[12]);

/* ------------------------------------------------------------------ pose algebra */

/* Combine local poses (mrw_trs arrays, as produced by clip sampling) before the hierarchy
 * compose - the blend / additive / mask layer. The engine owns the state machine; marrow gives
 * the 2-input primitives. Shared contract: out-param + mrw_result; zero-alloc; joint_count==0 ⇒
 * MRW_OK; joint_count>joint_capacity ⇒ MRW_E_CAPACITY; ANY non-finite input (pose component, mask,
 * weight) ⇒ MRW_E_RANGE with NO output written; `w`/`weight` and each `mask[j]` are CLAMPED to
 * [0,1] (not validated); in-place output MUST be the EXACT-SAME buffer as an input (partial overlap
 * is undefined). Input quaternions are assumed near-unit (clip samples already are). The batch
 * variants below produce identical results (SIMD within a small tolerance). */

/* Blend: out[j] = blend(a[j], b[j], we), we = clamp(w·(mask?mask[j]:1), 0,1): rot = hemisphere-
   corrected nlerp, trans/scale = lerp. mask NULL ⇒ uniform w. out MAY exactly alias a and/or b.
   we=0 ⇒ out[j]=a[j] (rotation as a rotation, q≡−q); we=1 ⇒ trans/scale=b[j], rot=b[j]. */
mrw_result mrw_pose_blend(const mrw_trs *a, const mrw_trs *b, float w,
                          const float *mask /*nullable, joint_count*/,
                          mrw_trs *out, uint32_t joint_count, uint32_t joint_capacity);

/* Build an additive (delta) pose of `pose` vs reference `base`: rot = canon(normalize(
   conj(base.rot)·pose.rot)) (canon ⇒ w≥0), trans = pose−base (identity 0), scale = pose⊘base RATIO
   (identity 1; a non-positive base or non-finite/non-positive ratio maps to 1). A non-identity
   delta is a DISTINCT SEMANTIC - NOT a valid local pose; never pass it to mrw_local_to_model.
   out_delta MAY exactly alias pose or base. */
mrw_result mrw_pose_make_additive(const mrw_trs *pose, const mrw_trs *base,
                                  mrw_trs *out_delta, uint32_t joint_count, uint32_t joint_capacity);

/* Apply `delta` on top of `base` at weight w + optional mask, we = clamp(w·(mask?mask[j]:1),
   0,1): rot = normalize(base.rot · nlerp_id(delta.rot, we)); trans = base + we·delta.trans;
   scale = base + we·(base·delta.scale − base) (linear toward pose, no pow). nlerp_id is
   nlerp-from-identity (no trig). Round-trip: accumulate(base, make_additive(pose,base), 1,
   NULL) ≡ pose - trans exact, scale exact for finite positive ratios, rotation up to sign (q≡−q).
   mask NULL ⇒ uniform w. out MAY exactly alias base. */
mrw_result mrw_pose_accumulate(const mrw_trs *base, const mrw_trs *delta, float w,
                               const float *mask /*nullable, joint_count*/,
                               mrw_trs *out, uint32_t joint_count, uint32_t joint_capacity);

/* ------------------------------------------------------------------ inverse kinematics */

/* Analytic single-chain IK that adjusts LOCAL rotations to meet a model-space goal (CPU only;
 * never batched). `model` MUST be the pre-IK model pose mrw_local_to_model produced from `locals`
 * (the consistency precondition): IK reads joint POSITIONS from model's translation columns and
 * derives world ROTATIONS by stripping UNIFORM scale from the model 3×3. The chain AND its
 * ancestors MUST be a proper, uniformly-scaled rotation (no shear / non-uniform scale / reflection)
 * - else MRW_E_UNSUPPORTED. IK writes back ONLY the chain's local rotations (trans/scale and other
 * joints untouched); the caller re-runs mrw_local_to_model over the affected subtree. Targets are in
 * model (joint-0) space. weight∈[0,1] (clamped) blends solved↔input (same rotation rule as
 * mrw_pose_blend); weight=0 is an exact no-op. Given the precondition, a non-finite value among the
 * inputs the solver consumes - the chain + ancestor `model` entries, the chain's local rotations,
 * target/pole/axes/weight - ⇒ MRW_E_RANGE with NO write (validated before the write-back; a
 * non-finite `locals` shows up in the matching `model` entry under the precondition). */

/* Two-bone analytic IK. Chain MUST be parent-linked (parent(mid_j)==root_j,
   parent(end_j)==mid_j; else MRW_E_INCOMPATIBLE). Bone lengths from the model positions; solves
   end_j's position onto target_model, bend plane oriented toward pole_model (a model-space
   POSITION); reach clamped to [|l1−l2|, l1+l2]. Degenerate cases (zero-length bone, target≈root,
   pole on the bone axis) have stable fallbacks (no-op or any-perpendicular bend). */
mrw_result mrw_ik_two_bone(const mrw_skeleton_view *skel, const float *model, mrw_trs *locals,
                           uint32_t root_j, uint32_t mid_j, uint32_t end_j,
                           const float target_model[3], const float pole_model[3],
                           float weight, uint32_t joint_capacity);

/* Aim / look-at. Rotates `joint` so aim_axis_local points at target_model (model POSITION);
   up_axis_local is rolled toward up_model (model DIRECTION) to fix twist. aim/up local axes MUST be
   non-collinear (else MRW_E_RANGE). Written back through the (scale-stripped) parent world rotation.
   target≈joint ⇒ no-op; up_model ∥ aim direction (or degenerate current up) ⇒ the roll is skipped. */
mrw_result mrw_ik_aim(const mrw_skeleton_view *skel, const float *model, mrw_trs *locals,
                      uint32_t joint, const float aim_axis_local[3], const float up_axis_local[3],
                      const float target_model[3], const float up_model[3],
                      float weight, uint32_t joint_capacity);

/* ------------------------------------------------------------------ batch */

/* Output palette component type. The layout - per-instance, joint-contiguous AoS 3×4, rows
 * row0,row1,row2, xyz=basis / w=translation - is identical for both; only the per-component
 * scalar narrows. F16 (binary16) halves the write/upload/fetch bandwidth at a precision cost (it is
 * opt-in; f32 is the default). uint32_t (not `typedef enum`) so the ABI width is fixed. */
typedef uint32_t mrw_palette_format;
enum { MRW_PALETTE_F32 = 0, MRW_PALETTE_F16 = 1 };

/* Homogeneous N-instance batch: animate `instance_count` instances that SHARE one skeleton,
 * one clip, and the output layout, each at its own time times[i] - the crowd CPU path. Output
 * is the per-instance, joint-contiguous AoS 3×4 palette: instance i, joint j occupies
 * out_palettes[(i*joint_count + j)*12 .. +12) (rows row0,row1,row2), each component a
 * binary32 (mrw_batch_clip_to_palette) or binary16 (mrw_batch_clip_to_palette_f16). The SCALAR
 * backend is bit-for-bit equal to looping mrw_clip_to_palette over the instances (and, for f16,
 * to mrw_f32_to_f16 of that result); SIMD backends match it within a small tolerance
 * (FMA/reassociation - the difference is visual-only). The SoA scratch is an internal
 * optimization (opaque layout) and stays binary32 regardless of out_format. No allocation. */

/* Scratch + output sizing. Either out-param MAY be NULL. Dispatch-independent: the chosen
 * backend never changes these. Size math is overflow-checked → MRW_E_OVERFLOW.
 *   out_format   : MRW_PALETTE_F32 / _F16 - sets the output element size (4 / 2 bytes); out of
 *                  range → MRW_E_RANGE. Scratch does NOT depend on it (scratch stays f32).
 *   out_scratch  : size + 64-byte (cache-line) alignment; depends only on joint_count.
 *   out_palettes : size + 16-byte alignment = instance_count*joint_count*12*elem (elem=4 F32/2 F16). */
mrw_result mrw_batch_clip_to_palette_requirements(
    uint32_t joint_count, uint32_t instance_count, mrw_palette_format out_format,
    mrw_mem_req *out_scratch, mrw_mem_req *out_palettes);

/* `disp` is validated: its `features` MUST be self-consistent with `backend` (else
 * MRW_E_UNSUPPORTED, no silent downgrade). SCALAR, SSE2, and AVX2(+FMA)(+F16C) are all implemented;
 * _detect picks the fastest the host supports. `scratch`
 * (>=64-aligned) and `out_palettes` (>=16-aligned) MUST meet the requirements query: smaller
 * → MRW_E_CAPACITY, misaligned → MRW_E_ALIGN. instance_count==0 is a clean no-op (the data
 * pointers may be NULL). Any non-finite times[i] → MRW_E_RANGE (no output written).
 * times / out_palettes / scratch MUST NOT overlap (caller precondition; violation is UB). */
mrw_result mrw_batch_clip_to_palette(
    const mrw_dispatch *disp,
    const mrw_skeleton_view *skel, const mrw_clip_view *clip,
    const float *times, uint32_t instance_count,
    float *out_palettes, size_t out_palettes_capacity,  /* bytes */
    void  *scratch,      size_t scratch_capacity);       /* bytes */

/* f16 (binary16) output twin of mrw_batch_clip_to_palette - same args, typed uint16_t* output (12
 * binary16 per joint entry; size from the requirements query with out_format=MRW_PALETTE_F16). The
 * f16 narrowing is the only difference; all validation, scratch, and dispatch behave identically -
 * EXCEPT alignment: the f16 scatter is fully unaligned, so out_palettes need only be 2-byte
 * (element) aligned, not 16. The requirements query still recommends a 16-byte base, but a caller
 * MAY write disjoint sub-ranges of one larger palette in place (out_palettes pointing mid-buffer at
 * the 24-B-per-joint stride - only 8-byte aligned on odd joint counts); smaller → MRW_E_ALIGN. */
mrw_result mrw_batch_clip_to_palette_f16(
    const mrw_dispatch *disp,
    const mrw_skeleton_view *skel, const mrw_clip_view *clip,
    const float *times, uint32_t instance_count,
    uint16_t *out_palettes, size_t out_palettes_capacity,  /* bytes */
    void     *scratch,      size_t scratch_capacity);       /* bytes */

/* ------------------------------------------------------------------ batch pose ops */

/* The blend/accumulate pose-combine math fused into the batch palette path for `instance_count`
 * instances that share one skeleton, the same source clip(s), and the output layout - the crowd
 * cross-fade / additive-layer. Identical contract to mrw_batch_clip_to_palette (AoS output;
 * caller-owned SoA scratch + _requirements() sizing; validated dispatch; f32 + f16 outputs;
 * instance_count==0 no-op; overflow-checked; same scratch/alignment/aliasing rules). The kernel
 * samples the source clip(s) into the SoA scratch, combines in component space (nlerp/lerp - no
 * trig, no pow), composes the hierarchy, and scatters the palette; the SCALAR backend is bit-for-
 * bit equal to looping the single-pose primitives (and, for f16, mrw_f32_to_f16 of that), SIMD
 * within a small tolerance. The requirements are dispatch-independent and equal the clip-batch sizes. */

/* Per instance i: blend(sample(clipA, timesA[i]), sample(clipB, timesB[i]), weights[i]) → palette i,
 * with an optional per-joint `mask` SHARED across the batch (joint_count, or NULL). clipA and clipB
 * MUST both pair with `skel` (matching ids; else MRW_E_INCOMPATIBLE). */
mrw_result mrw_batch_blend_clips_to_palette_requirements(
    uint32_t joint_count, uint32_t instance_count, mrw_palette_format out_format,
    mrw_mem_req *out_scratch, mrw_mem_req *out_palettes);
mrw_result mrw_batch_blend_clips_to_palette(
    const mrw_dispatch *disp, const mrw_skeleton_view *skel,
    const mrw_clip_view *clipA, const mrw_clip_view *clipB,
    const float *timesA, const float *timesB, const float *weights,
    const float *mask /*joint_count, or NULL*/, uint32_t instance_count,
    float *out_palettes, size_t out_palettes_capacity,  /* bytes */
    void  *scratch,      size_t scratch_capacity);       /* bytes */
mrw_result mrw_batch_blend_clips_to_palette_f16(
    const mrw_dispatch *disp, const mrw_skeleton_view *skel,
    const mrw_clip_view *clipA, const mrw_clip_view *clipB,
    const float *timesA, const float *timesB, const float *weights,
    const float *mask /*joint_count, or NULL*/, uint32_t instance_count,
    uint16_t *out_palettes, size_t out_palettes_capacity, /* bytes */
    void     *scratch,      size_t scratch_capacity);      /* bytes */

/* Per instance i: accumulate(sample(base_clip, times[i]), delta, weights[i], mask) → palette i,
 * where `delta` is a SINGLE shared additive pose (joint_count mrw_trs) applied at the
 * per-instance weight, with an optional per-joint `mask` shared across the batch. base_clip MUST
 * pair with `skel` (matching ids; else MRW_E_INCOMPATIBLE). */
mrw_result mrw_batch_accumulate_to_palette_requirements(
    uint32_t joint_count, uint32_t instance_count, mrw_palette_format out_format,
    mrw_mem_req *out_scratch, mrw_mem_req *out_palettes);
mrw_result mrw_batch_accumulate_to_palette(
    const mrw_dispatch *disp, const mrw_skeleton_view *skel, const mrw_clip_view *base_clip,
    const float *times, const mrw_trs *delta /*joint_count, shared*/,
    const float *weights, const float *mask /*joint_count, or NULL*/, uint32_t instance_count,
    float *out_palettes, size_t out_palettes_capacity,  /* bytes */
    void  *scratch,      size_t scratch_capacity);       /* bytes */
mrw_result mrw_batch_accumulate_to_palette_f16(
    const mrw_dispatch *disp, const mrw_skeleton_view *skel, const mrw_clip_view *base_clip,
    const float *times, const mrw_trs *delta /*joint_count, shared*/,
    const float *weights, const float *mask /*joint_count, or NULL*/, uint32_t instance_count,
    uint16_t *out_palettes, size_t out_palettes_capacity, /* bytes */
    void     *scratch,      size_t scratch_capacity);      /* bytes */

/* ------------------------------------------------------------------ root motion */

/* Accumulated root-motion delta D = U(t0)⁻¹·U(t1), applied as entity' = entity·D.
 * Times are unwrapped (monotonic clip time); any finite t is valid, non-finite ⇒
 * MRW_E_RANGE. Identity delta when sample_count==1, no HAS_ROOT_MOTION, or t0==t1.
 * out_delta is rigid (scale = 1). */
mrw_result mrw_root_motion(const mrw_clip_view *clip, float t0, float t1,
                                 mrw_xform *out_delta);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MRW_H */

/*
 * Copyright (c) 2026 Jérémie St-Amand
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */