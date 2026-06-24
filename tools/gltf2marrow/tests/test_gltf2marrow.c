/* gltf2marrow end-to-end test. Builds a small glTF in-memory (base64 buffers - no committed
 * binaries) that exercises the conversion hazards at once: skin.joints[] given out of order
 * (remap), a non-joint Armature wrapper above the root joint and a non-joint node between two
 * joints (the chain fold), inverse-bind remap + column→row major, and a LINEAR rotation
 * clip. The fixture's default pose equals its bind pose, so the rest-pose skinning palette must
 * be identity per joint - the strongest single gate (catches fold, remap, and major-order bugs). */
#include "convert.h"
#include "marrow.h"
#include "mrw_authoring.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MRW_G2M_TESTDIR
#define MRW_G2M_TESTDIR "."
#endif

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)
#define CHECK_NEAR(a,b,eps) CHECK(fabs((double)(a)-(double)(b)) <= (eps))

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void base64(const uint8_t *in, size_t n, char *out) {
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t b0 = in[i];
        uint32_t b1 = (i + 1 < n) ? in[i+1] : 0;
        uint32_t b2 = (i + 2 < n) ? in[i+2] : 0;
        uint32_t t = (b0 << 16) | (b1 << 8) | b2;
        out[o++] = B64[(t >> 18) & 63];
        out[o++] = B64[(t >> 12) & 63];
        out[o++] = (i + 1 < n) ? B64[(t >> 6) & 63] : '=';
        out[o++] = (i + 2 < n) ? B64[t & 63] : '=';
    }
    out[o] = '\0';
}

static void mat4_ident_trans(float m[16], float x, float y, float z) {
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
    m[12] = x; m[13] = y; m[14] = z;
}

/* affine ≈ identity 3×4? */
static int affine_is_ident(const float m[12], double eps) {
    static const float I[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    for (int i = 0; i < 12; ++i) if (fabs((double)m[i] - I[i]) > eps) return 0;
    return 1;
}

int main(void) {
    /* ---- buffer: 2 inverse-bind MAT4 (skin order [J1,J0]) + times[2] + rots[2] ---- */
    float fb[42];
    /* IBM[0] for J1 = inverse of bind global (13,1,2) → translate (-13,-1,-2) */
    mat4_ident_trans(fb + 0,  -13.0f, -1.0f, -2.0f);
    /* IBM[1] for J0 = inverse of bind global (10,1,0) → translate (-10,-1,0) */
    mat4_ident_trans(fb + 16, -10.0f, -1.0f,  0.0f);
    /* times */
    fb[32] = 0.0f; fb[33] = 1.0f;
    /* rotations: identity, then +90° about Y = (0, sin45, 0, cos45) */
    float s = (float)sqrt(0.5);
    fb[34] = 0; fb[35] = 0;  fb[36] = 0; fb[37] = 1;          /* key 0: identity */
    fb[38] = 0; fb[39] = s;  fb[40] = 0; fb[41] = s;          /* key 1: rotY90   */

    uint8_t bytes[168];
    memcpy(bytes, fb, sizeof bytes);
    char b64[256];
    base64(bytes, sizeof bytes, b64);

    char json[3072];
    snprintf(json, sizeof json,
        "{\"asset\":{\"version\":\"2.0\"},"
        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
        "\"nodes\":["
          "{\"name\":\"Armature\",\"translation\":[10,0,0],\"children\":[1]},"
          "{\"name\":\"J0\",\"translation\":[0,1,0],\"children\":[2]},"
          "{\"name\":\"mid\",\"translation\":[0,0,2],\"children\":[3]},"
          "{\"name\":\"J1\",\"translation\":[3,0,0]}"
        "],"
        "\"skins\":[{\"name\":\"rig\",\"skeleton\":1,\"joints\":[3,1],\"inverseBindMatrices\":0}],"
        "\"animations\":[{\"name\":\"spin\","
          "\"samplers\":[{\"input\":1,\"output\":2,\"interpolation\":\"LINEAR\"}],"
          "\"channels\":[{\"sampler\":0,\"target\":{\"node\":1,\"path\":\"rotation\"}}]}],"
        "\"buffers\":[{\"byteLength\":168,\"uri\":\"data:application/octet-stream;base64,%s\"}],"
        "\"bufferViews\":["
          "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":128},"
          "{\"buffer\":0,\"byteOffset\":128,\"byteLength\":8},"
          "{\"buffer\":0,\"byteOffset\":136,\"byteLength\":32}"
        "],"
        "\"accessors\":["
          "{\"bufferView\":0,\"componentType\":5126,\"count\":2,\"type\":\"MAT4\"},"
          "{\"bufferView\":1,\"componentType\":5126,\"count\":2,\"type\":\"SCALAR\",\"min\":[0],\"max\":[1]},"
          "{\"bufferView\":2,\"componentType\":5126,\"count\":2,\"type\":\"VEC4\"}"
        "]}", b64);

    const char *path = MRW_G2M_TESTDIR "/fixture_rig.gltf";
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    CHECK(fwrite(json, 1, strlen(json), f) == strlen(json));
    CHECK(fclose(f) == 0);

    /* ---- convert ---- */
    mrw_g2m_options opt = {0};
    opt.input_path = path; opt.fps = 30.0f; opt.skin_index = -1;
    uint8_t *buf = NULL; size_t size = 0; char diag[256] = {0};
    mrw_result r = mrw_g2m_convert(&opt, &buf, &size, diag, sizeof diag);
    if (r != MRW_OK) { fprintf(stderr, "convert failed: %s\n", diag); return 1; }

    /* ---- open + structural checks ---- */
    mrw_blob blob;
    CHECK(mrw_blob_open(buf, (uint64_t)size, &blob) == MRW_OK);
    mrw_skeleton_view sv;
    CHECK(mrw_blob_skeleton(&blob, &sv) == MRW_OK);
    CHECK(sv.joint_count == 2);

    uint16_t p0, p1; const char *n0, *n1;
    CHECK(mrw_skeleton_parent(&sv, 0, &p0) == MRW_OK && p0 == 0xFFFF); /* root = J0 (topo order) */
    CHECK(mrw_skeleton_parent(&sv, 1, &p1) == MRW_OK && p1 == 0);
    CHECK(mrw_skeleton_joint_name(&sv, 0, &n0) == MRW_OK && strcmp(n0, "J0") == 0);
    CHECK(mrw_skeleton_joint_name(&sv, 1, &n1) == MRW_OK && strcmp(n1, "J1") == 0);

    /* ---- rest-pose palette identity (fold + remap + inverse-bind together) ---- */
    mrw_trs locals[2];
    CHECK(mrw_skeleton_rest_local(&sv, 0, &locals[0]) == MRW_OK);
    CHECK(mrw_skeleton_rest_local(&sv, 1, &locals[1]) == MRW_OK);
    /* sanity on the folded translations */
    CHECK_NEAR(locals[0].trans[0], 10.0, 1e-5); CHECK_NEAR(locals[0].trans[1], 1.0, 1e-5); CHECK_NEAR(locals[0].trans[2], 0.0, 1e-5);
    CHECK_NEAR(locals[1].trans[0],  3.0, 1e-5); CHECK_NEAR(locals[1].trans[1], 0.0, 1e-5); CHECK_NEAR(locals[1].trans[2], 2.0, 1e-5);

    float *model   = (float *)mrw_authoring_alloc(2 * 12 * sizeof(float));
    float *palette = (float *)mrw_authoring_alloc(2 * 12 * sizeof(float));
    CHECK(model && palette);
    CHECK(mrw_local_to_model(&sv, locals, model, 2) == MRW_OK);
    CHECK(mrw_model_to_palette(&sv, model, palette, 2) == MRW_OK);
    CHECK(affine_is_ident(palette + 0,  1e-4));
    CHECK(affine_is_ident(palette + 12, 1e-4));

    /* ---- clip: duration-preserving fps + LINEAR rotation sampling ---- */
    mrw_clip_view cv;
    CHECK(mrw_clip_view_at(&blob, 1, &cv) == MRW_OK || mrw_clip_view_at(&blob, 2, &cv) == MRW_OK);
    CHECK(cv.sample_count == 31);                       /* floor(1.0*30+0.5)+1 */
    CHECK_NEAR(cv.fps, 30.0 / 1.0, 1e-3);               /* (sample_count-1)/D == D back-solved */

    mrw_trs s0, sN;
    CHECK(mrw_clip_sample(&cv, 0, 0, &s0) == MRW_OK);                  /* J0 @ t=0 → identity rot */
    CHECK(mrw_clip_sample(&cv, 0, cv.sample_count - 1, &sN) == MRW_OK);/* J0 @ t=D → rotY90 */
    CHECK_NEAR(s0.rot[1], 0.0, 1e-4);   CHECK_NEAR(s0.rot[3], 1.0, 1e-4);
    CHECK_NEAR(sN.rot[1], s,   1e-3);   CHECK_NEAR(sN.rot[3], s,   1e-3);
    /* folded translation (Armature offset) is constant across the spin */
    CHECK_NEAR(s0.trans[0], 10.0, 1e-4); CHECK_NEAR(sN.trans[0], 10.0, 1e-4);

    /* ---- determinism: a second conversion of the same input is byte-identical (ids included) ---- */
    uint8_t *buf2 = NULL; size_t size2 = 0; char diag2[256] = {0};
    CHECK(mrw_g2m_convert(&opt, &buf2, &size2, diag2, sizeof diag2) == MRW_OK);
    CHECK(size2 == size);
    CHECK(memcmp(buf, buf2, size) == 0);
    mrw_authoring_free(buf2);

    /* ---- duplicate --anim must dedup (one clip), not overflow the selection array ---- */
    const char *dup_names[2] = { "spin", "spin" };
    mrw_g2m_options dopt = opt; dopt.anims = dup_names; dopt.anim_count = 2;
    uint8_t *dbuf = NULL; size_t dsize = 0; char ddiag[256] = {0};
    CHECK(mrw_g2m_convert(&dopt, &dbuf, &dsize, ddiag, sizeof ddiag) == MRW_OK);
    mrw_blob dblob; CHECK(mrw_blob_open(dbuf, (uint64_t)dsize, &dblob) == MRW_OK);
    uint32_t nclips = 0;
    for (uint32_t i = 0; i < dblob.section_count; ++i) { uint32_t ty = 0; if (mrw_blob_section_type(&dblob, i, &ty) == MRW_OK && ty == MRW_SECTION_CLIP) ++nclips; }
    CHECK(nclips == 1);
    mrw_authoring_free(dbuf);

    /* ---- absurd fps must be rejected cleanly (no float→uint UB, no giant allocation) ---- */
    mrw_g2m_options hopt = opt; hopt.fps = 1e30f;
    uint8_t *hbuf = NULL; size_t hsize = 0; char hdiag[256] = {0};
    CHECK(mrw_g2m_convert(&hopt, &hbuf, &hsize, hdiag, sizeof hdiag) != MRW_OK);
    CHECK(hbuf == NULL);

    mrw_authoring_free(model); mrw_authoring_free(palette);
    mrw_authoring_free(buf);
    printf("test_gltf2marrow: OK\n");
    return 0;
}
