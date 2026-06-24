/* Time/duration/frame sampling boundaries: t=0, t=duration, just-below-loop,
 * negative t, count==1. Uses a 1-joint clip whose translation.x ramps with frame index
 * (fps=1 ⇒ duration=count-1 ⇒ sampled trans.x == t_local). */
#include "test_util.h"
#include "mrw_build.h"
#include <string.h>

static const uint16_t PARENT[1] = { 0xFFFF };
static const char *NAMES[1] = { "root" };
static const float REST[10] = { 0,0,0,1, 0,0,0, 1,1,1 };
static const float IB[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };

/* build a 1-joint ramp clip: sample s has trans.x = s (+ optional base) */
static void make_samples(float *dst, uint32_t sc, float base) {
    for (uint32_t s = 0; s < sc; ++s) {
        float trs[10] = { 0,0,0,1, base + (float)s, 0, 0, 1, 1, 1 };
        memcpy(dst + (size_t)s*10, trs, 40);
    }
}

static float sample_x(const mrw_clip_view *cv, float t) {
    mrw_trs l;
    mrw_clip_sample_local(cv, t, &l, 1);
    return l.trans[0];
}

int main(void) {
    float sA[5*10], sB[5*10], sC[1*10];
    make_samples(sA, 5, 0.0f);   /* non-looping ramp 0..4 */
    make_samples(sB, 5, 0.0f);   /* looping ramp 0..4     */
    make_samples(sC, 1, 7.0f);   /* static, x=7           */

    mrw_skel skel = { 1, PARENT, REST, IB, NAMES };
    mrw_clip clips[3] = {
        { 1.0f, 5, 0,                    sA, NULL },
        { 1.0f, 5, MRW_CLIP_LOOPING,  sB, NULL },
        { 1.0f, 1, 0,                    sC, NULL },
    };
    uint8_t *buf = NULL;
    size_t sz = mrw_build(&skel, clips, 3, NULL, &buf);
    mrw_blob blob;
    CHECK_EQ(mrw_blob_open(buf, sz, &blob), MRW_OK);

    mrw_clip_view nonloop, loop, stat;
    CHECK_EQ(mrw_clip_view_at(&blob, 1, &nonloop), MRW_OK);
    CHECK_EQ(mrw_clip_view_at(&blob, 2, &loop), MRW_OK);
    CHECK_EQ(mrw_clip_view_at(&blob, 3, &stat), MRW_OK);

    /* ---- non-looping (duration 4) ---- */
    CHECK_NEAR(sample_x(&nonloop, 0.0f), 0.0f, 1e-5);   /* t=0          */
    CHECK_NEAR(sample_x(&nonloop, 2.5f), 2.5f, 1e-5);   /* midpoint     */
    CHECK_NEAR(sample_x(&nonloop, 4.0f), 4.0f, 1e-5);   /* t=duration   */
    CHECK_NEAR(sample_x(&nonloop, -1.0f), 0.0f, 1e-5);  /* clamp low    */
    CHECK_NEAR(sample_x(&nonloop, 10.0f), 4.0f, 1e-5);  /* clamp high   */

    /* ---- looping (duration 4) ---- */
    CHECK_NEAR(sample_x(&loop, 0.0f), 0.0f, 1e-5);
    CHECK_NEAR(sample_x(&loop, 4.0f), 0.0f, 1e-5);      /* t=duration wraps to 0 */
    CHECK_NEAR(sample_x(&loop, 3.5f), 3.5f, 1e-5);      /* just below loop       */
    CHECK_NEAR(sample_x(&loop, 3.999f), 3.999f, 1e-3);  /* just below loop       */
    CHECK_NEAR(sample_x(&loop, -1.0f), 3.0f, 1e-5);     /* negative wraps        */
    CHECK_NEAR(sample_x(&loop, 8.0f), 0.0f, 1e-5);      /* multi-period wrap     */

    /* ---- static (count==1) ---- */
    CHECK_NEAR(sample_x(&stat, 0.0f), 7.0f, 1e-5);
    CHECK_NEAR(sample_x(&stat, 100.0f), 7.0f, 1e-5);
    CHECK_NEAR(sample_x(&stat, -100.0f), 7.0f, 1e-5);

    /* non-finite t rejected */
    mrw_trs l;
    CHECK_EQ(mrw_clip_sample_local(&nonloop, (float)NAN, &l, 1), MRW_E_RANGE);
    CHECK_EQ(mrw_clip_sample_local(&nonloop, (float)INFINITY, &l, 1), MRW_E_RANGE);

    mrw_free(buf);
    printf(g_fail ? "test_sampling: %d FAILED\n" : "test_sampling: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
