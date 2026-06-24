#include "mrw_build.h"
#include "marrow_internal.h" /* LE readers for the blob parse helpers */

#include <stdio.h>
#include <stdlib.h>

/* Trusted-input wrapper: tests build only well-formed blobs, so a rejection here is a test bug -
 * abort loudly with the result code rather than silently returning 0. */
size_t mrw_build(const mrw_skel *skel, const mrw_clip *clips, uint32_t nclip,
                 const mrw_baked *baked, uint8_t **out_buf) {
    uint8_t *buf = NULL; size_t size = 0;
    mrw_result r = mrw_authoring_build(skel, clips, nclip, baked, &buf, &size);
    if (r != MRW_OK) {
        fprintf(stderr, "mrw_build: authoring rejected trusted input (mrw_result=%d)\n", (int)r);
        abort();
    }
    *out_buf = buf;
    return size;
}

void     mrw_free(uint8_t *buf) { mrw_authoring_free(buf); }
uint8_t *mrw_alloc64(size_t n)  { return (uint8_t *)mrw_authoring_alloc(n); }
uint16_t mrw_f32_to_half(float f) { return mrw_authoring_f32_to_half(f); }

/* ---- blob parse helpers ---- */
uint32_t mrw_blob_section_count(const uint8_t *buf) { return mrw_rd_u32(buf + 16); }
uint64_t mrw_blob_section_offset(const uint8_t *buf, uint32_t index) {
    uint32_t st = mrw_rd_u32(buf + 20);
    return mrw_rd_u64(buf + st + (size_t)index * 32 + 8);
}
uint32_t mrw_blob_section_type_raw(const uint8_t *buf, uint32_t index) {
    uint32_t st = mrw_rd_u32(buf + 20);
    return mrw_rd_u32(buf + st + (size_t)index * 32);
}
