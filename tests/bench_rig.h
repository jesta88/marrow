/* Deterministic rig generator shared by the batch parity test and the batch benchmark.
 * Builds a chain skeleton of `jc` joints + a dense clip of `sc` samples with pseudo-random
 * (but reproducible) rotations/translations, so the pose pipeline does real work. */
#ifndef BENCH_RIG_H
#define BENCH_RIG_H

#include "test_util.h"   /* compute_bind_inverse, math, mrw_trs/affine helpers */
#include "mrw_build.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define RIG_MAX_JOINTS  68u   /* covers the heavy-rig parity shapes (65/67) */
#define RIG_MAX_SAMPLES 256u
#define RIG_FPS         30.0f

/* tiny deterministic LCG (no rand(): identical sequence on every platform) */
typedef struct { uint32_t s; } rig_rng;
static uint32_t rig_u32(rig_rng *r) { r->s = r->s * 1664525u + 1013904223u; return r->s; }
static float    rig_f(rig_rng *r, float lo, float hi) {
    return lo + (hi - lo) * ((float)(rig_u32(r) >> 8) / (float)(1u << 24)); /* [lo,hi) */
}

/* unit quaternion from a random axis-angle (always near-unit ⇒ passes loader validation) */
static void rig_quat(rig_rng *r, float q[4]) {
    float ax = rig_f(r, -1, 1), ay = rig_f(r, -1, 1), az = rig_f(r, -1, 1);
    float n = sqrtf(ax*ax + ay*ay + az*az);
    if (n < 1e-6f) { ax = 0; ay = 0; az = 1; n = 1; }
    float ang = rig_f(r, -3.14159265f, 3.14159265f), s = sinf(ang * 0.5f);
    q[0] = ax/n * s; q[1] = ay/n * s; q[2] = az/n * s; q[3] = cosf(ang * 0.5f);
}

/* Build a {skeleton, clip} blob into *out_buf (free with mrw_free). `clip_flags` accepts
 * MRW_CLIP_LOOPING (then ensure sc>=2). Clip duration is (sc-1)/RIG_FPS. `codec` selects the on-disk
 * clip encoding (0 = raw TRS, 1 = scale-free q4+t3); the rig is unit-scale so codec 1 is exact. */
static size_t build_rig_blob_codec(uint32_t jc, uint32_t sc, uint32_t clip_flags, uint32_t seed,
                                   uint32_t codec, uint8_t **out_buf) {
    static uint16_t parent[RIG_MAX_JOINTS];
    static float    rest[RIG_MAX_JOINTS * 10];
    static float    ib[RIG_MAX_JOINTS * 12];
    static char     namebuf[RIG_MAX_JOINTS][8];
    static const char *names[RIG_MAX_JOINTS];
    static float    samples[RIG_MAX_JOINTS * RIG_MAX_SAMPLES * 10];

    rig_rng r = { seed ? seed : 1u };
    for (uint32_t j = 0; j < jc; ++j) {
        parent[j] = (j == 0) ? 0xFFFFu : (uint16_t)(j - 1);   /* chain (parent < j) */
        float q[4]; rig_quat(&r, q);
        rest[j*10+0] = q[0]; rest[j*10+1] = q[1]; rest[j*10+2] = q[2]; rest[j*10+3] = q[3];
        rest[j*10+4] = (j == 0) ? 0.0f : rig_f(&r, 0.2f, 1.2f); /* offset child along +x-ish */
        rest[j*10+5] = rig_f(&r, -0.3f, 0.3f);
        rest[j*10+6] = rig_f(&r, -0.3f, 0.3f);
        rest[j*10+7] = rest[j*10+8] = rest[j*10+9] = 1.0f;     /* unit scale */
        snprintf(namebuf[j], sizeof namebuf[j], "j%u", j);
        names[j] = namebuf[j];
    }
    compute_bind_inverse(jc, parent, rest, ib);

    for (uint32_t j = 0; j < jc; ++j) {
        for (uint32_t s = 0; s < sc; ++s) {
            float *smp = samples + ((size_t)j * sc + s) * 10;
            float q[4]; rig_quat(&r, q);
            smp[0] = q[0]; smp[1] = q[1]; smp[2] = q[2]; smp[3] = q[3];
            smp[4] = rig_f(&r, -1.0f, 1.0f);
            smp[5] = rig_f(&r, -1.0f, 1.0f);
            smp[6] = rig_f(&r, -1.0f, 1.0f);
            smp[7] = smp[8] = smp[9] = 1.0f;
        }
    }

    mrw_skel skel = { jc, parent, rest, ib, names };
    mrw_clip clip = { RIG_FPS, sc, clip_flags, samples, NULL, codec };
    return mrw_build(&skel, &clip, 1, NULL, out_buf);
}

/* codec-0 convenience wrapper (the common case for callers that don't vary the codec). */
static size_t build_rig_blob(uint32_t jc, uint32_t sc, uint32_t clip_flags, uint32_t seed,
                             uint8_t **out_buf) {
    return build_rig_blob_codec(jc, sc, clip_flags, seed, 0u, out_buf);
}

#endif /* BENCH_RIG_H */
