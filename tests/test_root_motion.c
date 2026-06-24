/* Root-motion property tests: forward/reverse, single/multi-loop wrap,
 * interval composition D(t0,t2)=D(t0,t1)·D(t1,t2), loop-boundary continuity, edges. */
#include "test_util.h"
#include "mrw_build.h"
#include <string.h>

static const uint16_t PARENT[1] = { 0xFFFF };
static const char *NAMES[1] = { "root" };
static const float REST[10] = { 0,0,0,1, 0,0,0, 1,1,1 };
static const float IB[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };
/* body pose: 1 joint, 2 identity samples (root motion ignores the body) */
static const float BODY[2*10] = { 0,0,0,1,0,0,0,1,1,1,  0,0,0,1,0,0,0,1,1,1 };

/* root trajectories (count=2): q4,t3 per sample */
static const float TRACK_TRANS[2*7] = { 0,0,0,1, 0,0,0,   0,0,0,1, 1,0,0 };       /* +x by 1 / cycle */
static float TRACK_ROT[2*7];                                                       /* +90° z / cycle  */

static mrw_xform D(const mrw_clip_view *cv, float t0, float t1) {
    mrw_xform d; CHECK_EQ(mrw_root_motion(cv, t0, t1, &d), MRW_OK); return d;
}

/* compose two rigid xforms as a 3x4 product (A∘B), for the telescoping check */
static void compose_check(const mrw_xform *a, const mrw_xform *b, const mrw_xform *ab) {
    float A[12], B[12], AB[12], prod[12];
    mrw_xform_to_affine(a, A);
    mrw_xform_to_affine(b, B);
    mrw_xform_to_affine(ab, AB);
    mrw_affine_mul(A, B, prod);             /* A·B == compose(A,B) */
    CHECK(aff_near(prod, AB, 1e-4));
}

int main(void) {
    float s45 = (float)sin(M_PI/4), c45 = (float)cos(M_PI/4);
    float rot[2*7] = { 0,0,0,1, 0,0,0,   0,0,s45,c45, 0,0,0 };
    memcpy(TRACK_ROT, rot, sizeof rot);

    mrw_skel skel = { 1, PARENT, REST, IB, NAMES };
    uint32_t RM = MRW_CLIP_HAS_ROOT_MOTION;
    mrw_clip clips[5] = {
        { 1.0f, 2, RM,                          BODY, TRACK_TRANS }, /* 1: trans non-loop */
        { 1.0f, 2, RM | MRW_CLIP_LOOPING,    BODY, TRACK_TRANS }, /* 2: trans loop     */
        { 1.0f, 2, RM,                          BODY, TRACK_ROT   }, /* 3: rot non-loop   */
        { 1.0f, 2, 0,                           BODY, NULL        }, /* 4: no root motion */
        { 1.0f, 1, 0,                           REST, NULL        }, /* 5: count==1       */
    };
    uint8_t *buf = NULL;
    size_t sz = mrw_build(&skel, clips, 5, NULL, &buf);
    mrw_blob blob;
    CHECK_EQ(mrw_blob_open(buf, sz, &blob), MRW_OK);

    mrw_clip_view tnl, tl, rnl, nort, one;
    CHECK_EQ(mrw_clip_view_at(&blob, 1, &tnl), MRW_OK);
    CHECK_EQ(mrw_clip_view_at(&blob, 2, &tl), MRW_OK);
    CHECK_EQ(mrw_clip_view_at(&blob, 3, &rnl), MRW_OK);
    CHECK_EQ(mrw_clip_view_at(&blob, 4, &nort), MRW_OK);
    CHECK_EQ(mrw_clip_view_at(&blob, 5, &one), MRW_OK);

    /* ---- translation, non-looping ---- */
    CHECK_NEAR(D(&tnl, 0, 1).trans[0], 1.0f, 1e-5);
    CHECK_NEAR(D(&tnl, 0, 0.5f).trans[0], 0.5f, 1e-5);
    CHECK_NEAR(D(&tnl, 0, 1.5f).trans[0], 1.0f, 1e-5);     /* clamp (no accumulation) */
    CHECK_NEAR(D(&tnl, 1, 0).trans[0], -1.0f, 1e-5);       /* reverse                 */
    { mrw_xform d01 = D(&tnl, 0, 0.3f), d12 = D(&tnl, 0.3f, 0.8f), d02 = D(&tnl, 0, 0.8f);
      compose_check(&d01, &d12, &d02); }

    /* ---- translation, looping (accumulates per cycle) ---- */
    CHECK_NEAR(D(&tl, 0, 2.5f).trans[0], 2.5f, 1e-4);      /* multi-loop */
    CHECK_NEAR(D(&tl, 0, -1.0f).trans[0], -1.0f, 1e-4);    /* negative   */
    CHECK_NEAR(D(&tl, 0.99f, 1.01f).trans[0], 0.02f, 1e-3);/* loop-boundary continuity */
    /* tiny negative time must stay near identity, not jump a whole −1 cycle: this is the
     * exact (k,tau) rounding case where t - floor(t/T)*T rounds up to exactly T. */
    CHECK_NEAR(D(&tl, 0, -1e-20f).trans[0], 0.0f, 1e-5);
    CHECK_NEAR(D(&tl, 0, -1e-7f).trans[0], -1e-7f, 1e-5);
    { mrw_xform d01 = D(&tl, 0.5f, 1.7f), d12 = D(&tl, 1.7f, 3.2f), d02 = D(&tl, 0.5f, 3.2f);
      compose_check(&d01, &d12, &d02); }

    /* ---- rotation, non-looping ---- */
    { mrw_xform d = D(&rnl, 0, 1); float m[12], R[9], q[4] = {0,0,s45,c45};
      mrw_xform_to_affine(&d, m); mrw_quat_to_mat3(q, R);
      float er[12] = { R[0],R[1],R[2],0, R[3],R[4],R[5],0, R[6],R[7],R[8],0 };
      CHECK(aff_near(m, er, 1e-4)); }
    { mrw_xform d01 = D(&rnl, 0, 0.5f), d12 = D(&rnl, 0.5f, 1.0f), d02 = D(&rnl, 0, 1.0f);
      compose_check(&d01, &d12, &d02); }

    /* ---- edges: identity deltas ---- */
    mrw_xform d;
    CHECK_EQ(mrw_root_motion(&tnl, 0.5f, 0.5f, &d), MRW_OK);   /* t0==t1 */
    CHECK_NEAR(d.rot[0], 0.0f, 1e-6); CHECK_NEAR(d.rot[1], 0.0f, 1e-6);
    CHECK_NEAR(d.rot[2], 0.0f, 1e-6); CHECK_NEAR(d.rot[3], 1.0f, 1e-6);
    CHECK_NEAR(d.trans[0], 0.0f, 1e-6); CHECK_NEAR(d.scale, 1.0f, 0);

    CHECK_EQ(mrw_root_motion(&nort, 0, 1, &d), MRW_OK);        /* no root motion */
    CHECK_NEAR(d.rot[3], 1.0f, 1e-6); CHECK_NEAR(d.trans[0], 0.0f, 1e-6);
    CHECK_EQ(mrw_root_motion(&one, 0, 1, &d), MRW_OK);         /* count==1 */
    CHECK_NEAR(d.rot[3], 1.0f, 1e-6); CHECK_NEAR(d.trans[0], 0.0f, 1e-6);

    /* non-finite ⇒ RANGE */
    CHECK_EQ(mrw_root_motion(&tnl, (float)NAN, 0, &d), MRW_E_RANGE);
    CHECK_EQ(mrw_root_motion(&tnl, 0, (float)INFINITY, &d), MRW_E_RANGE);

    mrw_free(buf);
    printf(g_fail ? "test_root_motion: %d FAILED\n" : "test_root_motion: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
