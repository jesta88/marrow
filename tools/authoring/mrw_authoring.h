/* Checked .mrw authoring for the offline tools. The marrow runtime only CONSUMES and
 * validates blobs - authoring/baking are offline, out of the runtime target - so the byte
 * emitter lives here, in tools/. It is the exact inverse of the loader: it lays out the file
 * header, section table, and SKELETON/CLIP/BAKED sections and stamps in the section identities.
 *
 * This API is CHECKED for arbitrary authoring input: every offset/size product is computed in
 * 64-bit and bounds-checked, count limits are enforced, and allocation failure is reported (never
 * aborted). */
#ifndef MRW_AUTHORING_H
#define MRW_AUTHORING_H

#include "marrow.h"

/* One skeleton. Arrays are joint-indexed in final marrow order (parent[j] < j). */
typedef struct {
    uint32_t joint_count;          /* [1, 0xFFFE]                        */
    const uint16_t *parent;        /* joint_count (0xFFFF = no parent)   */
    const float    *rest_local;    /* joint_count * 10 (q4,t3,s3)        */
    const float    *inverse_bind;  /* joint_count * 12 (3x4 row-major)   */
    const char    **names;         /* joint_count NUL-terminated strings */
} mrw_skel;

/* One dense fixed-rate clip. `samples` is ALWAYS the joint-major 10-float (q4+t3+s3)
 * layout; `codec` selects the ON-DISK encoding: 0 = raw TRS (40 B/sample, all 10 floats); 1 =
 * scale-free (28 B/sample, q4+t3 only - emit only when every sample's scale is exactly 1.0f, which
 * the authoring front-end snaps after an eligibility test, so the clip id over the snapped 10-float
 * stream equals the codec-1 canonical id). `codec` last so existing positional initializers (codec 0)
 * stay valid. */
typedef struct {
    float    fps;
    uint32_t sample_count;
    uint32_t flags;                /* MRW_CLIP_LOOPING | MRW_CLIP_HAS_ROOT_MOTION */
    const float *samples;          /* joint_count * sample_count * 10 (joint-major) */
    const float *root_track;       /* sample_count * 7, or NULL  */
    uint32_t codec;                /* 0 = raw TRS (default), 1 = scale-free (q4+t3) */
} mrw_clip;

/* One baked GPU section. Produced by marrow-bake; not used by gltf2marrow. */
typedef struct {
    uint32_t frame_stride_texels;  /* 0 => bone_count*2 */
    uint32_t total_frames;
    const uint16_t *texels;        /* total_frames * frame_stride_texels * 4 raw halves */
    uint32_t clip_count;
    const uint32_t *clip_index;    /* -> index into clips[] */
    const uint32_t *first_frame;
    const uint32_t *frame_count;
    const float    *source_duration;
    const uint32_t *clip_flags;    /* MRW_BAKED_CLIP_LOOPING */
} mrw_baked;

/* Build a blob into a freshly 64-aligned allocation. `skel` may be NULL (no skeleton); `baked`
 * may be NULL. Section order: SKELETON (if any), then clips, then BAKED (if any). On MRW_OK,
 * *out_buf holds the blob (free with mrw_authoring_free) and *out_size its byte length; on any
 * error nothing is allocated. Returns:
 *   MRW_E_FORMAT    a required pointer is NULL, or a flag implies missing data (HAS_ROOT_MOTION
 *                   without a root_track);
 *   MRW_E_RANGE     a count is out of range (joint_count not in [1,0xFFFE], sample_count 0,
 *                   non-finite/non-positive fps where duration needs it, LOOPING with <2 samples);
 *   MRW_E_OVERFLOW  size/offset arithmetic would overflow the wire fields, OR the blob is too
 *                   large to allocate (allocation failed). */
mrw_result mrw_authoring_build(const mrw_skel *skel,
                               const mrw_clip *clips, uint32_t nclip,
                               const mrw_baked *baked,
                               uint8_t **out_buf, size_t *out_size);

/* 64-aligned alloc/free for blobs (and for tests that duplicate blobs). Tool-owned - NOT part
 * of the runtime ABI; the runtime never allocates. */
void    *mrw_authoring_alloc(size_t n);
void     mrw_authoring_free(void *p);

/* IEEE-754 binary32 -> binary16 (round-to-nearest-even); for authoring baked texels. */
uint16_t mrw_authoring_f32_to_half(float f);

#endif /* MRW_AUTHORING_H */
