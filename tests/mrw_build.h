/* Test-only thin wrappers over the checked authoring library (mrw_authoring.{h,c}, in
 * tools/authoring), which owns the mrw_skel/mrw_clip/mrw_baked structs and the byte emitter.
 * These wrappers present a trusted-input API (size_t return, abort-on-failure) for the test call
 * sites. Also exposes a few read-only blob parse helpers used by the malformed-fixture mutation
 * tests. */
#ifndef MRW_BUILD_H
#define MRW_BUILD_H

#include "mrw_authoring.h"   /* mrw_skel / mrw_clip / mrw_baked, mrw_authoring_alloc/free/f32_to_half */

/* Build a blob into a freshly 64-aligned allocation (trusted test input - aborts if the checked
 * authoring build rejects it). Returns the blob size and stores the buffer in *out_buf (free it
 * with mrw_free). Section order: SKELETON (if any), then clips, then BAKED (if any). */
size_t mrw_build(const mrw_skel *skel,
                 const mrw_clip *clips, uint32_t nclip,
                 const mrw_baked *baked,
                 uint8_t **out_buf);

void     mrw_free(uint8_t *buf);
uint8_t *mrw_alloc64(size_t n);  /* 64-aligned alloc (for duplicating blobs in tests) */

/* IEEE-754 binary32 -> binary16 (round-to-nearest-even); for authoring baked texels. */
uint16_t mrw_f32_to_half(float f);

/* Parse helpers over a built blob (for malformed-fixture mutation, pre-open). */
uint32_t mrw_blob_section_count(const uint8_t *buf);
uint64_t mrw_blob_section_offset(const uint8_t *buf, uint32_t index);
uint32_t mrw_blob_section_type_raw(const uint8_t *buf, uint32_t index);

#endif /* MRW_BUILD_H */
