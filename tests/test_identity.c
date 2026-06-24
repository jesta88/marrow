/* Content-identity tests: the library's FNV-1a-128 over the canonical
 * stream must equal the independent Python reference's golden vectors, and a built blob's
 * stored ids must equal the recomputed-from-view ids. */
#include "test_util.h"
#include "mrw_build.h"
#include "golden_ids.h"
#include <string.h>

/* Must mirror tests/oracle_ids.py exactly. */
static const uint16_t PARENT[2] = { 0xFFFF, 0 };
static const float REST[2*10] = {
    0,0,0,1, 0,0,0, 1,1,1,
    0,0,0,1, 1,0,0, 1,1,1,
};
static const float IB[2*12] = {
    1,0,0,0, 0,1,0,0, 0,0,1,0,
    1,0,0,-1, 0,1,0,0, 0,0,1,0,
};
static const char *NAMES[2] = { "a", "b" };

static const float SAMP[2*2*10] = {
    /* joint-major: j0s0, j0s1, j1s0, j1s1 */
    0,0,0,1, 0,0,0, 1,1,1,
    0,0,0,1, 0,0,0, 1,1,1,
    0,0,0,1, 1,0,0, 1,1,1,
    0,0,0,1, 1,0,0, 1,1,1,
};

int main(void) {
    /* name blob for raw skeleton id: "a\0b\0" with offsets 0,2 */
    uint8_t name_blob[4] = { 'a', 0, 'b', 0 };
    uint32_t name_off[2] = { 0, 2 };

    mrw_id128 sid;
    CHECK_EQ(mrw_skeleton_compute_id(2, 4, PARENT, REST, IB, name_off, name_blob, &sid), MRW_OK);
    mrw_id128 golden_s = { GOLDEN_SKELETON_ID_LO, GOLDEN_SKELETON_ID_HI };
    CHECK(mrw_id_equal(&sid, &golden_s));

    mrw_id128 cid;
    CHECK_EQ(mrw_clip_compute_id(2, 2.0f, 2, MRW_CLIP_LOOPING, SAMP, NULL, &cid), MRW_OK);
    mrw_id128 golden_c = { GOLDEN_CLIP_ID_LO, GOLDEN_CLIP_ID_HI };
    CHECK(mrw_id_equal(&cid, &golden_c));

    /* Build a blob from the same data; stored ids must equal recomputed-from-view ids,
     * and the clip's skeleton_id must resolve to the skeleton. */
    mrw_skel skel = { 2, PARENT, REST, IB, NAMES };
    mrw_clip clip = { 2.0f, 2, MRW_CLIP_LOOPING, SAMP, NULL };
    uint8_t *buf = NULL;
    size_t sz = mrw_build(&skel, &clip, 1, NULL, &buf);
    mrw_blob blob;
    CHECK_EQ(mrw_blob_open(buf, sz, &blob), MRW_OK);

    mrw_skeleton_view sv; mrw_clip_view cv;
    CHECK_EQ(mrw_blob_skeleton(&blob, &sv), MRW_OK);
    CHECK_EQ(mrw_clip_view_at(&blob, 1, &cv), MRW_OK);
    CHECK(mrw_id_equal(&sv.id, &golden_s));
    CHECK(mrw_id_equal(&cv.id, &golden_c));
    CHECK(mrw_id_equal(&cv.skeleton_id, &sv.id));

    mrw_id128 rv;
    mrw_skeleton_view_id(&sv, &rv); CHECK(mrw_id_equal(&rv, &sv.id));
    mrw_clip_view_id(&cv, &rv);     CHECK(mrw_id_equal(&rv, &cv.id));

    /* lookup by id */
    mrw_skeleton_view sv2;
    CHECK_EQ(mrw_blob_skeleton_by_id(&blob, &golden_s, &sv2), MRW_OK);
    mrw_clip_view cv2;
    CHECK_EQ(mrw_blob_clip_by_id(&blob, &golden_c, &cv2), MRW_OK);

    mrw_free(buf);
    printf(g_fail ? "test_identity: %d FAILED\n" : "test_identity: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
