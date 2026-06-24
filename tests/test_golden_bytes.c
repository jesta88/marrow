/* Golden byte-level fixture: the in-memory build must be byte-identical to
 * the committed fixtures/rig.mrw, and that committed blob must load and view correctly.
 * If the wire layout intentionally changes, regenerate with the gen_fixtures target. */
#include "test_util.h"
#include "fixture_rig.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MRW_FIXTURE_DIR
#define MRW_FIXTURE_DIR "."
#endif

int main(void) {
    uint8_t *mem = NULL;
    size_t sz = build_fixture(&mem);

    char path[1024];
    snprintf(path, sizeof path, "%s/rig.mrw", MRW_FIXTURE_DIR);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL: cannot open committed fixture %s\n", path); mrw_free(mem); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *disk = mrw_alloc64((size_t)fsz);
    size_t got = fread(disk, 1, (size_t)fsz, f);
    fclose(f);
    CHECK_EQ(got, (size_t)fsz);

    /* byte-identical */
    CHECK_EQ((size_t)fsz, sz);
    if ((size_t)fsz == sz) CHECK_EQ(memcmp(disk, mem, sz), 0);

    /* the committed blob loads and views correctly */
    mrw_blob blob;
    CHECK_EQ(mrw_blob_open(disk, (uint64_t)fsz, &blob), MRW_OK);
    CHECK_EQ(blob.section_count, 3);

    mrw_skeleton_view sv; mrw_clip_view cv; mrw_baked_view bv;
    CHECK_EQ(mrw_blob_skeleton(&blob, &sv), MRW_OK);
    CHECK_EQ(sv.joint_count, 3);
    const char *nm; mrw_skeleton_joint_name(&sv, 2, &nm); CHECK(strcmp(nm, "head") == 0);

    CHECK_EQ(mrw_clip_view_at(&blob, 1, &cv), MRW_OK);
    CHECK_EQ(cv.sample_count, 2);
    CHECK(mrw_id_equal(&cv.skeleton_id, &sv.id));

    CHECK_EQ(mrw_baked_view_at(&blob, 2, &bv), MRW_OK);
    CHECK_EQ(bv.encoding, 1); CHECK_EQ(bv.bone_count, 3); CHECK_EQ(bv.texels_per_bone, 2);
    CHECK(mrw_id_equal(&bv.skeleton_id, &sv.id));
    /* decode a baked bone (identity texels ⇒ identity affine) */
    float aff[12];
    CHECK_EQ(mrw_baked_sample_bone(&bv, 0, 1, 0.0f, aff), MRW_OK);
    CHECK(aff_near(aff, MRW_IDENTITY12, 1e-3));

    mrw_free(disk); mrw_free(mem);
    printf(g_fail ? "test_golden_bytes: %d FAILED\n" : "test_golden_bytes: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
