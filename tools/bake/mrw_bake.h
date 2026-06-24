/* Baked GPU bake core (offline). The marrow runtime is consume-only: it decodes/validates baked
 * blobs but never produces them (baking lives in the offline tool, not the runtime ABI). This is
 * that bake: CPU palette → decompose to Q+T+uniform-scale → binary16 texel stream + per-rig
 * eligibility verdict. The matrix↔TRS decompose math it builds on lives in the authoring lib
 * (tools/authoring/mrw_decompose); this core adds only the baking-specific texel encode/decode and
 * temporal sampling, reusing the runtime scalar math (mrw_clip_to_palette / mrw_xform_to_affine /
 * mrw_half_to_float). It is allocation-free: all buffers are caller-owned and sized by
 * mrw_bake_clip_requirements (BYO-allocator, no runtime malloc). The front-end (bake_run) and the
 * runtime parity check both link this core; it never pulls cgltf or an allocator. */
#ifndef MRW_BAKE_H
#define MRW_BAKE_H

#include "marrow.h"
#include "mrw_decompose.h"  /* mrw_mat3_to_quat, mrw_decompose_affine, mrw_affine_apply,
                             * mrw_decompose_residual, mrw_affine_probe_dist */

/* ---- component-space decode of a raw baked texel stream ----
 * These operate on the half stream the baker emits (frame_stride_texels = bone_count·2), NOT
 * via the runtime's mrw_baked_sample_bone (which returns the COMPOSED affine). Cross-fade must
 * interpolate in Q/T/s space BEFORE composing, so the parity check needs the pre-compose
 * form these expose. */

/* Decode one bone at one absolute frame into (q renormalized, t, s). */
void mrw_bake_decode(const uint16_t *texels, uint32_t frame_stride_texels,
                     uint32_t frame, uint32_t bone, mrw_xform *out);

/* Temporal sample (frame index + component nlerp/lerp) of one bone at clip-local
 * time t, in component space. `first_frame`..`first_frame+frame_count` is the bone's frame
 * window; source_duration / looping per the baked clip-table entry. */
void mrw_bake_sample_xform(const uint16_t *texels, uint32_t frame_stride_texels,
                           uint32_t first_frame, uint32_t frame_count, float source_duration,
                           int looping, uint32_t bone, float t, mrw_xform *out);

/* Cross-clip blend of two component-space xforms (≤2-clip cross-fade): hemisphere-
 * corrected quaternion nlerp + linear t/s lerp. The sanctioned baked-tier blend. */
void mrw_xform_nlerp(const mrw_xform *a, const mrw_xform *b, float w, mrw_xform *out);

/* ---- bake one clip into a caller-owned texel buffer ---- */

/* Why the worst bone of a clip failed (for diagnostics). MRW_BAKE_OK when the clip is eligible. */
typedef enum {
    MRW_BAKE_OK = 0,        /* eligible (the worst probed bone is within tol)              */
    MRW_BAKE_STRUCTURAL,    /* a bone's skinning transform does not polar-decompose         */
    MRW_BAKE_RESIDUAL,      /* a probed bone's reconstruction residual exceeds tol         */
    MRW_BAKE_QUANTIZED      /* a bone's encoded texel decodes non-finite (binary16 overflow)*/
} mrw_bake_reason;

/* Per-clip eligibility verdict + the worst offender, for the front-end's diagnostics. */
typedef struct {
    int             eligible;      /* 1 = every bone passed on every frame of this clip            */
    float           max_residual;  /* worst probe displacement (m); +INF if any probed bone is
                                    * structural (or a texel quantizes to non-finite)             */
    uint32_t        worst_bone;    /* bone index of the worst offender                            */
    uint32_t        worst_frame;   /* baked frame index of the worst offender                     */
    mrw_bake_reason reason;        /* why the worst bone failed (MRW_BAKE_OK iff eligible)         */
} mrw_bake_stats;

/* Size BOTH the scratch and the per-clip output texel buffer (mirrors *_requirements).
 * Either out-param MAY be NULL. Size math is overflow-checked → MRW_E_OVERFLOW; frame_count == 0 ⇒
 * MRW_E_RANGE. Both regions need 16-byte alignment.
 *   out_scratch : depends only on joint_count = joint_count · (12+12+4)·sizeof(float)
 *                 (a 16-aligned model + palette region for mrw_clip_to_palette + a per-bone quat
 *                  region for sign tracking).
 *   out_texels  : frame_count · joint_count · 2 texels · 8 bytes (RGBA16F, 2 texels/bone). */
mrw_result mrw_bake_clip_requirements(uint32_t joint_count, uint32_t frame_count,
                                      mrw_mem_req *out_scratch, mrw_mem_req *out_texels);

/* Bake ONE clip into out_texels: `frame_count` endpoint-inclusive frames (==1 ⇒ static),
 * each bone_count·2 texels (frame_stride = bone_count·2). Per bone, the temporal quaternion track
 * is made sign-continuous: frame f's quat is flipped if dot(q_{f−1}, q_f) < 0. Baked frame f
 * samples the CPU palette (mrw_clip_to_palette) at t_f = f/(frame_count−1)·duration, decomposes each
 * bone's palette entry to Q+T+s, and quantizes to binary16.
 *
 * probe_counts[b] = #probes for bone b (NULL ⇒ all zero ⇒ every bone eligible by default for the
 * perceptual test). probes is the flat, per-bone-contiguous (Σ counts)·3 float array in bind space
 * (NULL allowed iff every count is 0). scratch/out_texels MUST meet mrw_bake_clip_requirements and
 * be 16-aligned and non-overlapping: smaller ⇒ MRW_E_CAPACITY, misaligned ⇒ MRW_E_ALIGN,
 * frame_count == 0 ⇒ MRW_E_RANGE.
 *
 * Texels are ALWAYS written when MRW_OK is returned (so an inspection / parity read is valid).
 * ELIGIBILITY is reported via *out_stats and is NOT an error - MRW_OK is returned whenever the
 * texels are written and the verdict is set. An error code is returned for a NULL skel/clip/scratch/
 * out_texels, probes==NULL with a positive count, a bad buffer (MRW_E_CAPACITY/ALIGN/RANGE), or a
 * CPU sampling failure (which a validated blob never hits); on error *out_stats is still set
 * (ineligible, +INF). Two rejections apply to EVERY bone regardless of probe count: a non-decomposable
 * palette (MRW_BAKE_STRUCTURAL) and an encoded texel that decodes non-finite (MRW_BAKE_QUANTIZED;
 * the loader rejects any non-finite texel) - "eligible by default" (no probes) exempts a bone only from
 * the perceptual residual. A single ineligible (bone, frame) rejects the whole clip. */
mrw_result mrw_bake_clip(const mrw_skeleton_view *skel, const mrw_clip_view *clip,
                         uint32_t frame_count, const uint32_t *probe_counts, const float *probes,
                         float tol, void *scratch, size_t scratch_capacity,
                         uint16_t *out_texels, size_t out_texels_capacity, mrw_bake_stats *out_stats);

#endif /* MRW_BAKE_H */
