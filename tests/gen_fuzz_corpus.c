/* Emits the seed corpus for the .mrw loader fuzzer. Run once and commit the output under
 * tests/fuzz/corpus/:
 *   gen_fuzz_corpus <out-dir>
 * Each seed is a VALID blob of a distinct shape, so libFuzzer (and the portable fuzz_replay coverage
 * gate) start from inputs that already pass the magic/version/section validation and exercise the
 * deep per-section checks instead of bouncing off the header. The shapes mirror the coverage the
 * fuzz_replay --check-coverage gate asserts: skeleton-only, codec-0 clip, codec-1 clip, root-motion
 * clip, single-clip baked, multi-clip baked.
 *
 * (Kept separate from gen_fixtures, which emits THE golden byte-identical fixture - corpus churn must
 * never threaten that gate.) */
#include "mrw_build.h"
#include "fixture_rig.h"
#include "bench_rig.h"
#include <stdio.h>
#include <string.h>

/* A small 3-joint rig shared by the hand-authored seeds. inverse_bind is only finite-checked by the
 * loader, so plain identity-ish 3x4 rows suffice; rest quats are exactly unit. */
static const uint16_t PARENT[3] = { 0xFFFF, 0, 1 };
static const char *NAMES[3] = { "root", "spine", "head" };
static const float REST[3*10] = {
    0,0,0,1, 0,0,0, 1,1,1,
    0,0,0,1, 1,0,0, 1,1,1,
    0,0,0,1, 0,1,0, 1,1,1,
};
static const float IB[3*12] = {
    1,0,0,0,  0,1,0,0,  0,0,1,0,
    1,0,0,-1, 0,1,0,0,  0,0,1,0,
    1,0,0,-1, 0,1,0,-1, 0,0,1,0,
};

static int dump(const char *dir, const char *name, const uint8_t *buf, size_t sz) {
    /* A seed must be a VALID blob - self-validate before committing it. */
    mrw_blob b;
    if (mrw_blob_open(buf, sz, &b) != MRW_OK) {
        fprintf(stderr, "gen_fuzz_corpus: %s does not validate - not a usable seed\n", name);
        return 1;
    }
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    size_t w = fwrite(buf, 1, sz, f);
    fclose(f);
    if (w != sz) { fprintf(stderr, "short write %s\n", path); return 1; }
    printf("wrote %s (%zu bytes)\n", path, sz);
    return 0;
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : ".";
    int rc = 0;
    uint8_t *buf = NULL;
    size_t sz;

    /* 1. skeleton-only (no clip, no baked) */
    {
        mrw_skel skel = { 3, PARENT, REST, IB, NAMES };
        sz = mrw_build(&skel, NULL, 0, NULL, &buf);
        rc |= dump(dir, "seed_skeleton_only.mrw", buf, sz);
        mrw_free(buf);
    }

    /* 2/3. skeleton + one clip, codec 0 and codec 1 (unit-scale rig ⇒ codec-1 exact) */
    sz = build_rig_blob_codec(8, 4, 0u, 7u, 0u, &buf);
    rc |= dump(dir, "seed_clip_codec0.mrw", buf, sz);
    mrw_free(buf);
    sz = build_rig_blob_codec(8, 4, 0u, 7u, 1u, &buf);
    rc |= dump(dir, "seed_clip_codec1.mrw", buf, sz);
    mrw_free(buf);

    /* 4. clip with root motion (exercises the root-track validation + reader) */
    {
        static const float SAMP[3*2*10] = {
            0,0,0,1, 0,0,0, 1,1,1,   0,0,0,1, 0,0,0, 1,1,1,
            0,0,0,1, 1,0,0, 1,1,1,   0,0,0,1, 1,0,0, 1,1,1,
            0,0,0,1, 0,1,0, 1,1,1,   0,0,0,1, 0,1,0, 1,1,1,
        };
        static const float ROOT[2*7] = {
            0,0,0,1, 0,0,0,
            0,0,0,1, 1,0,0,
        };
        mrw_skel skel = { 3, PARENT, REST, IB, NAMES };
        mrw_clip clip = { 1.0f, 2, MRW_CLIP_HAS_ROOT_MOTION, SAMP, ROOT, 0 };
        sz = mrw_build(&skel, &clip, 1, NULL, &buf);
        rc |= dump(dir, "seed_rootmotion.mrw", buf, sz);
        mrw_free(buf);
    }

    /* 5. single-clip baked (the golden fixture shape: skeleton + clip + BAKED) */
    sz = build_fixture(&buf);
    rc |= dump(dir, "seed_baked.mrw", buf, sz);
    mrw_free(buf);

    /* 6. multi-clip baked: skeleton + 2 clips + a BAKED referencing both (clip_count==2) */
    {
        static float SAMP_A[3*2*10], SAMP_B[3*2*10];
        for (int i = 0; i < 3*2; ++i) {
            float *a = SAMP_A + i*10, *b = SAMP_B + i*10;
            a[0]=a[1]=a[2]=0; a[3]=1; a[4]=a[5]=a[6]=0;    a[7]=a[8]=a[9]=1;
            b[0]=b[1]=b[2]=0; b[3]=1; b[4]=0.5f; b[5]=b[6]=0; b[7]=b[8]=b[9]=1; /* differs ⇒ distinct id */
        }
        static uint16_t tex[24*4];   /* total_frames(4) * stride(bone_count*2=6) = 24 texels */
        for (int i = 0; i < 24; ++i) {
            tex[i*4+0] = 0; tex[i*4+1] = 0; tex[i*4+2] = 0; tex[i*4+3] = mrw_f32_to_half(1.0f);
        }
        static const uint32_t bidx[2] = {0,1}, bff[2] = {0,2}, bfc[2] = {2,2}, bflags[2] = {0,0};
        static const float bdur[2] = { 1.0f, 1.0f };
        mrw_skel skel = { 3, PARENT, REST, IB, NAMES };
        mrw_clip clips[2] = {
            { 1.0f, 2, 0, SAMP_A, NULL, 0 },
            { 1.0f, 2, 0, SAMP_B, NULL, 0 },
        };
        mrw_baked baked = { 0, 4, tex, 2, bidx, bff, bfc, bdur, bflags };
        sz = mrw_build(&skel, clips, 2, &baked, &buf);
        rc |= dump(dir, "seed_baked_multiclip.mrw", buf, sz);
        mrw_free(buf);
    }

    return rc;
}
