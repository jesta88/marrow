/* Loader views + pose pipeline. Builds a 3-joint rig, validates
 * the views, and checks the skinning palette against closed-form expectations. */
#include "test_util.h"
#include "mrw_build.h"
#include <string.h>
#include <stdlib.h>

static const uint16_t PARENT[3] = { 0xFFFF, 0, 1 };
static const char *NAMES[3] = { "root", "j1", "j2" };

/* rest_local: root@origin, j1 at +x, j2 at +x of j1 (chain along +x) */
static const float REST[3*10] = {
    0,0,0,1,  0,0,0,  1,1,1,
    0,0,0,1,  1,0,0,  1,1,1,
    0,0,0,1,  1,0,0,  1,1,1,
};

int main(void) {
    float ib[3*12];
    compute_bind_inverse(3, PARENT, REST, ib);

    mrw_skel skel = { 3, PARENT, REST, ib, NAMES };

    /* clip: 2 samples; s0 == rest (bind), s1 rotates the ROOT 90° about +Z */
    float s45 = (float)sin(M_PI/4), c45 = (float)cos(M_PI/4);
    float samples[3*2*10];
    /* joint-major: index (j*2 + s)*10 */
    memcpy(samples + (0*2+0)*10, REST + 0*10, 40);
    float root_rot[10] = { 0,0,s45,c45, 0,0,0, 1,1,1 };
    memcpy(samples + (0*2+1)*10, root_rot, 40);
    memcpy(samples + (1*2+0)*10, REST + 1*10, 40);
    memcpy(samples + (1*2+1)*10, REST + 1*10, 40);
    memcpy(samples + (2*2+0)*10, REST + 2*10, 40);
    memcpy(samples + (2*2+1)*10, REST + 2*10, 40);

    mrw_clip clip = { /*fps*/1.0f, /*sc*/2, /*flags*/0, samples, NULL };

    uint8_t *buf = NULL;
    size_t sz = mrw_build(&skel, &clip, 1, NULL, &buf);

    mrw_blob blob;
    CHECK_EQ(mrw_blob_open(buf, sz, &blob), MRW_OK);
    CHECK_EQ(blob.section_count, 2);

    /* ---- skeleton view ---- */
    mrw_skeleton_view sv;
    CHECK_EQ(mrw_blob_skeleton(&blob, &sv), MRW_OK);
    CHECK_EQ(sv.joint_count, 3);
    uint16_t p;
    mrw_skeleton_parent(&sv, 0, &p); CHECK_EQ(p, 0xFFFF);
    mrw_skeleton_parent(&sv, 2, &p); CHECK_EQ(p, 1);
    const char *nm;
    mrw_skeleton_joint_name(&sv, 1, &nm); CHECK(strcmp(nm, "j1") == 0);
    mrw_trs rl;
    mrw_skeleton_rest_local(&sv, 1, &rl);
    CHECK_NEAR(rl.trans[0], 1, 0); CHECK_NEAR(rl.scale[2], 1, 0);
    /* recomputed id matches stored id */
    mrw_id128 rid;
    mrw_skeleton_view_id(&sv, &rid);
    CHECK(mrw_id_equal(&rid, &sv.id));

    /* ---- clip view ---- */
    mrw_clip_view cv;
    CHECK_EQ(mrw_clip_view_at(&blob, 1, &cv), MRW_OK);
    CHECK_EQ(cv.joint_count, 3); CHECK_EQ(cv.sample_count, 2); CHECK_EQ(cv.codec, 0);
    CHECK(mrw_id_equal(&cv.skeleton_id, &sv.id));
    mrw_id128 cid; mrw_clip_view_id(&cv, &cid); CHECK(mrw_id_equal(&cid, &cv.id));

    float scratch[3*12], palette[3*12];

    /* t=0: pose == bind ⇒ palette is identity for every joint */
    CHECK_EQ(mrw_clip_to_palette(&sv, &cv, 0.0f, scratch, palette, 3), MRW_OK);
    for (int j = 0; j < 3; ++j) CHECK(aff_near(palette + j*12, MRW_IDENTITY12, 1e-5));

    /* fused path == staged path (sample → local→model → palette) */
    mrw_trs locals[3]; float model[3*12], palette2[3*12];
    CHECK_EQ(mrw_clip_sample_local(&cv, 0.0f, locals, 3), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv, locals, model, 3), MRW_OK);
    CHECK_EQ(mrw_model_to_palette(&sv, model, palette2, 3), MRW_OK);
    for (int i = 0; i < 3*12; ++i) CHECK_NEAR(palette[i], palette2[i], 1e-6);

    /* capacity is enforced */
    CHECK_EQ(mrw_clip_to_palette(&sv, &cv, 0.0f, scratch, palette, 2), MRW_E_CAPACITY);
    CHECK_EQ(mrw_clip_sample_local(&cv, 0.0f, locals, 2), MRW_E_CAPACITY);

    /* t=duration: whole chain rotated 90° about origin ⇒ every palette == [R|0] */
    float R[9]; float q[4] = { 0,0,s45,c45 }; mrw_quat_to_mat3(q, R);
    float expectR[12] = { R[0],R[1],R[2],0, R[3],R[4],R[5],0, R[6],R[7],R[8],0 };
    CHECK_EQ(mrw_clip_to_palette(&sv, &cv, 1.0f, scratch, palette, 3), MRW_OK);
    for (int j = 0; j < 3; ++j) CHECK(aff_near(palette + j*12, expectR, 1e-5));

    mrw_free(buf);
    printf(g_fail ? "test_format_pose: %d FAILED\n" : "test_format_pose: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
