/* codec-1 (scale-free q4+t3) gates beyond the randomized parity in test_batch:
 *  (1) round-trip + scale semantics - a codec-1 clip opens, reports codec 1, samples scale ≡ 1, and
 *      its q/t bytes equal the codec-0 clip's; the codec-1 blob is smaller (−30% sample bytes).
 *  (2) id equivalence - for a BIT-EXACT-1.0f unit clip, id(codec0) == id(codec1) (codec is not hashed;
 *      the canonical stream is q4+t3+(1,1,1) either way). The design makes no claim for non-exact scales.
 *  (3) bake parity - bake(codec0) and bake(codec1) baked-tier texels are byte-identical for a unit clip.
 *  (4) malformed codec-1 - a truncated 28-B sample span is rejected (span_ok ⇒ MRW_E_RANGE). */
#include "test_util.h"
#include "bench_rig.h"
#include "marrow_internal.h"   /* mrw_rd_*, wire offsets */
#include "mrw_bake.h"
#include <stdlib.h>

static void test_roundtrip_and_id(uint32_t jc, uint32_t sc, uint32_t flags) {
    uint8_t *b0 = NULL, *b1 = NULL;
    size_t sz0 = build_rig_blob_codec(jc, sc, flags, 7u, 0u, &b0);
    size_t sz1 = build_rig_blob_codec(jc, sc, flags, 7u, 1u, &b1);  /* SAME rig+seed, different codec */
    mrw_blob bl0, bl1;
    CHECK_EQ(mrw_blob_open(b0, sz0, &bl0), MRW_OK);
    CHECK_EQ(mrw_blob_open(b1, sz1, &bl1), MRW_OK);
    mrw_clip_view c0, c1;
    CHECK_EQ(mrw_clip_view_at(&bl0, 1, &c0), MRW_OK);
    CHECK_EQ(mrw_clip_view_at(&bl1, 1, &c1), MRW_OK);

    CHECK(c0.codec == 0u);
    CHECK(c1.codec == 1u);
    CHECK(sz1 < sz0);  /* 28 vs 40 B/sample ⇒ smaller blob */

    /* id equivalence: bit-exact 1.0f scales ⇒ identical canonical stream ⇒ identical id. */
    mrw_id128 id0, id1;
    CHECK_EQ(mrw_clip_view_id(&c0, &id0), MRW_OK);
    CHECK_EQ(mrw_clip_view_id(&c1, &id1), MRW_OK);
    CHECK(mrw_id_equal(&id0, &id1));
    CHECK(mrw_id_equal(&c0.id, &id0));   /* stored id matches the recomputed one (both codecs) */
    CHECK(mrw_id_equal(&c1.id, &id1));

    /* per-sample: codec-1 returns scale ≡ 1 and the same q/t as codec-0. */
    for (uint32_t j = 0; j < jc; ++j)
        for (uint32_t s = 0; s < sc; ++s) {
            mrw_trs a, b;
            CHECK_EQ(mrw_clip_sample(&c0, j, s, &a), MRW_OK);
            CHECK_EQ(mrw_clip_sample(&c1, j, s, &b), MRW_OK);
            for (int k = 0; k < 4; ++k) CHECK(a.rot[k]   == b.rot[k]);
            for (int k = 0; k < 3; ++k) CHECK(a.trans[k] == b.trans[k]);
            CHECK(b.scale[0] == 1.0f && b.scale[1] == 1.0f && b.scale[2] == 1.0f);
        }
    mrw_free(b0); mrw_free(b1);
}

static void test_bake_parity(uint32_t jc, uint32_t sc) {
    uint8_t *b0 = NULL, *b1 = NULL;
    size_t sz0 = build_rig_blob_codec(jc, sc, 0u, 9u, 0u, &b0);
    size_t sz1 = build_rig_blob_codec(jc, sc, 0u, 9u, 1u, &b1);
    mrw_blob bl0, bl1;
    CHECK_EQ(mrw_blob_open(b0, sz0, &bl0), MRW_OK);
    CHECK_EQ(mrw_blob_open(b1, sz1, &bl1), MRW_OK);
    mrw_skeleton_view s0, s1; mrw_clip_view c0, c1;
    CHECK_EQ(mrw_blob_skeleton(&bl0, &s0), MRW_OK); CHECK_EQ(mrw_clip_view_at(&bl0, 1, &c0), MRW_OK);
    CHECK_EQ(mrw_blob_skeleton(&bl1, &s1), MRW_OK); CHECK_EQ(mrw_clip_view_at(&bl1, 1, &c1), MRW_OK);

    uint32_t fc = sc; /* bake fc endpoint-inclusive frames */
    mrw_mem_req sreq, treq;
    CHECK_EQ(mrw_bake_clip_requirements(jc, fc, &sreq, &treq), MRW_OK);
    uint8_t *scr  = mrw_alloc64(sreq.size);
    uint16_t *t0  = (uint16_t *)mrw_alloc64(treq.size);
    uint16_t *t1  = (uint16_t *)mrw_alloc64(treq.size);
    mrw_bake_stats st0, st1;
    CHECK_EQ(mrw_bake_clip(&s0, &c0, fc, NULL, NULL, 1e-2f, scr, sreq.size, t0, treq.size, &st0), MRW_OK);
    CHECK_EQ(mrw_bake_clip(&s1, &c1, fc, NULL, NULL, 1e-2f, scr, sreq.size, t1, treq.size, &st1), MRW_OK);
    /* unit clip ⇒ codec-0 and codec-1 sample to the same poses ⇒ identical baked-tier texels. */
    CHECK(memcmp(t0, t1, treq.size) == 0);
    mrw_free(scr); mrw_free((uint8_t *)t0); mrw_free((uint8_t *)t1);
    mrw_free(b0); mrw_free(b1);
}

static void test_malformed_truncated(void) {
    uint32_t jc = 8u, sc = 8u;
    uint8_t *b1 = NULL; size_t sz1 = build_rig_blob_codec(jc, sc, 0u, 3u, 1u, &b1);
    mrw_blob bl; CHECK_EQ(mrw_blob_open(b1, sz1, &bl), MRW_OK);   /* sanity: valid as authored */

    /* Shrink the CLIP section's size (section table entry 1, +16 u64) to header-only (128 B): the
     * 28-B/sample span no longer fits ⇒ span_ok rejects with MRW_E_RANGE. */
    uint8_t *c = mrw_alloc64(sz1); memcpy(c, b1, sz1);
    uint32_t st = mrw_rd_u32(c + 20);          /* section table offset */
    memcpy(c + st + 32u * 1u + 16u, &(uint64_t){ MRW_SECTION_HEADER_SIZE }, 8);
    mrw_blob _b; CHECK_EQ(mrw_blob_open(c, sz1, &_b), MRW_E_RANGE);
    mrw_free(c); mrw_free(b1);
}

int main(void) {
    test_roundtrip_and_id(8, 8, MRW_CLIP_LOOPING);
    test_roundtrip_and_id(65, 5, 0);
    test_roundtrip_and_id(67, 1, 0);   /* static */
    test_bake_parity(20, 8);
    test_malformed_truncated();
    printf(g_fail ? "test_codec1: %d FAILED\n" : "test_codec1: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
