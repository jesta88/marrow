/* Coverage-guided fuzz target for the .mrw loader.
 *
 * mrw_blob_open is marrow's ONLY untrusted-input boundary: the whole "validate, then view"
 * contract rests on it rejecting every malformed input cleanly AND on the post-validation
 * accessors never reading out of bounds. This harness feeds arbitrary bytes to mrw_blob_open and,
 * on success, exercises the full PUBLIC read surface (marrow.h) so ASan/UBSan trap any over-read or
 * UB the validation pass missed. It is a black-box test of the loader contract - only marrow.h is
 * used, never the internal wire helpers.
 *
 * The library this links against is built with -fsanitize=fuzzer-no-link so libFuzzer gets edge
 * coverage INSIDE mrw_blob_open and the validators (CMake target `marrow_fuzz`); instrumenting only
 * this TU would leave the fuzzer coverage-blind. */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L   /* posix_memalign under strict -std=c11 */
#endif
#include "fuzz_blob.h"
#include "marrow.h"
#include <string.h>

/* Exact-size 64-byte-aligned allocation so the sanitizer redzone abuts base+size: a loader
 * over-read by even one byte traps, instead of landing inside addressable over-allocation slack.
 * These are TEST allocations - not the runtime, which never allocates; banned_scan scopes to
 * src/ + the bake core, not tests/. */
#if defined(_WIN32)
#  include <malloc.h>
static void *fz_alloc(size_t n) { return _aligned_malloc(n, 64); }
static void  fz_free(void *p)   { _aligned_free(p); }
#else
#  include <stdlib.h>
static void *fz_alloc(size_t n) { void *p = NULL; return posix_memalign(&p, 64, n) ? NULL : p; }
static void  fz_free(void *p)   { free(p); }
#endif

/* Walk a validated SKELETON: re-read the whole stream (id recompute), the by-id lookup, and every
 * joint-indexed reader at joint 0..joint_count INCLUSIVE - index==joint_count is the out-of-range
 * boundary that MUST return MRW_E_RANGE without a read. */
static void walk_skeleton(const mrw_blob *b, const mrw_skeleton_view *v) {
    mrw_id128 id;
    (void)mrw_skeleton_view_id(v, &id);                 /* re-reads the entire skeleton stream */
    mrw_skeleton_view tmp;
    (void)mrw_blob_skeleton_by_id(b, &v->id, &tmp);
    for (uint32_t j = 0; j <= v->joint_count; ++j) {
        uint16_t parent; (void)mrw_skeleton_parent(v, j, &parent);
        mrw_trs trs;     (void)mrw_skeleton_rest_local(v, j, &trs);
        float ib[12];    (void)mrw_skeleton_inverse_bind(v, j, ib);
        const char *name = NULL;
        if (mrw_skeleton_joint_name(v, j, &name) == MRW_OK && name) {
            /* read the bytes - the loader guaranteed a NUL within the name blob */
            volatile char sink = 0;
            for (const char *s = name; *s; ++s) sink = *s;
            (void)sink;
        }
    }
}

/* Walk a validated CLIP: id recompute (re-reads the whole sample stream), by-id lookup, every
 * (joint, sample) including the boundaries, the root-sample reader, and the root-motion delta. */
static void walk_clip(const mrw_blob *b, const mrw_clip_view *v) {
    mrw_id128 id;
    (void)mrw_clip_view_id(v, &id);                     /* re-reads the entire sample stream */
    mrw_clip_view tmp;
    (void)mrw_blob_clip_by_id(b, &v->id, &tmp);
    for (uint32_t j = 0; j <= v->joint_count; ++j)
        for (uint32_t s = 0; s <= v->sample_count; ++s) {
            mrw_trs trs; (void)mrw_clip_sample(v, j, s, &trs);
        }
    for (uint32_t s = 0; s <= v->sample_count; ++s) {
        mrw_xform xf; (void)mrw_clip_root_sample(v, s, &xf);
    }
    float dur = (v->sample_count > 1 && v->fps > 0.0f)
                    ? (float)(v->sample_count - 1) / v->fps : 1.0f;
    mrw_xform delta;
    (void)mrw_root_motion(v, 0.0f, dur, &delta);
    (void)mrw_root_motion(v, 0.0f, dur * 0.5f, &delta);
}

/* Walk a validated BAKED: every clip-table entry (incl. the boundary), and mrw_baked_sample_bone
 * over every (clip, bone) at frame endpoints + an interior + a past-end phase. The decode reader is
 * the highest-risk accessor (frame-indexed texel fetch). */
static void walk_baked(const mrw_baked_view *v) {
    for (uint32_t ci = 0; ci <= v->clip_count; ++ci) {
        mrw_baked_clip e;
        if (mrw_baked_clip_entry(v, ci, &e) != MRW_OK) continue;   /* skips ci==clip_count */
        float dur = e.source_duration_s;
        const float ts[4] = { 0.0f, dur * 0.5f, dur, dur * 1.5f }; /* endpoints, interior, past-end */
        for (uint32_t bone = 0; bone <= v->bone_count; ++bone)
            for (int k = 0; k < 4; ++k) {
                float out12[12];
                (void)mrw_baked_sample_bone(v, ci, bone, ts[k], out12);
            }
    }
    /* The clip loop skips ci==clip_count (mrw_baked_clip_entry returns early), so the decoder's OWN
     * clip_index/bone range checks are never boundary-probed there - hit them directly. */
    float out12[12];
    (void)mrw_baked_sample_bone(v, v->clip_count, 0, 0.0f, out12);              /* clip_index OOB */
    (void)mrw_baked_sample_bone(v, 0, v->bone_count, 0.0f, out12);              /* bone OOB       */
}

static void walk_blob(const mrw_blob *b) {
    for (uint32_t i = 0; i <= b->section_count; ++i) {  /* i==section_count = boundary */
        uint32_t type;
        (void)mrw_blob_section_type(b, i, &type);
        mrw_skeleton_view sk;
        if (mrw_skeleton_view_at(b, i, &sk) == MRW_OK) walk_skeleton(b, &sk);
        mrw_clip_view cl;
        if (mrw_clip_view_at(b, i, &cl) == MRW_OK) walk_clip(b, &cl);
        mrw_baked_view bk;
        if (mrw_baked_view_at(b, i, &bk) == MRW_OK) walk_baked(&bk);
    }
    mrw_skeleton_view sk;
    (void)mrw_blob_skeleton(b, &sk);                    /* the at-most-one convenience lookup */
}

int mrw_fuzz_one(const uint8_t *data, size_t size) {
    size_t cap = size ? size : 1;       /* size==0 still needs a valid non-NULL aligned ptr */
    uint8_t *buf = (uint8_t *)fz_alloc(cap);
    if (!buf) return 0;                 /* allocator failure: nothing to test, don't abort */
    if (size) memcpy(buf, data, size);

    mrw_blob blob;
    if (mrw_blob_open(buf, (uint64_t)size, &blob) == MRW_OK)
        walk_blob(&blob);

    fz_free(buf);
    return 0;
}

/* libFuzzer entry. Defined unconditionally: in the libFuzzer target, libFuzzer supplies main and
 * calls this; in the portable replay target (fuzz_blob.c + fuzz_replay.c) it is simply an unused
 * function - fuzz_replay.c owns main - so there is no two-main conflict and no fragile "is libFuzzer
 * linked" macro. */
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) { return mrw_fuzz_one(data, size); }
