/* Math core tests (quaternion, affine, cofactor, half decode). */
#include "test_util.h"
#include <string.h>

static void test_quat_identity(void) {
    float q[4] = { 0, 0, 0, 1 }, r[9];
    mrw_quat_to_mat3(q, r);
    float expect[9] = { 1,0,0, 0,1,0, 0,0,1 };
    for (int i = 0; i < 9; ++i) CHECK_NEAR(r[i], expect[i], 1e-6);
}

static void test_quat_z90(void) {
    /* 90° about +Z: x->y, y->-x */
    float c = (float)cos(M_PI / 4), s = (float)sin(M_PI / 4);
    float q[4] = { 0, 0, s, c }, r[9];
    mrw_quat_to_mat3(q, r);
    float vx[3] = { r[0], r[3], r[6] }; /* R * (1,0,0) = first column */
    CHECK_NEAR(vx[0], 0, 1e-5); CHECK_NEAR(vx[1], 1, 1e-5); CHECK_NEAR(vx[2], 0, 1e-5);
    float vy[3] = { r[1], r[4], r[7] }; /* R * (0,1,0) */
    CHECK_NEAR(vy[0], -1, 1e-5); CHECK_NEAR(vy[1], 0, 1e-5); CHECK_NEAR(vy[2], 0, 1e-5);
}

static void test_trs_compose(void) {
    /* scale columns then translate: p' = T + R*(S*p) */
    mrw_trs trs;
    float q[4] = { 0, 0, 0, 1 };
    memcpy(trs.rot, q, sizeof q);
    trs.trans[0] = 5; trs.trans[1] = 6; trs.trans[2] = 7;
    trs.scale[0] = 2; trs.scale[1] = 3; trs.scale[2] = 4;
    float m[12];
    mrw_trs_to_affine(&trs, m);
    /* apply to (1,1,1): expect (2+5, 3+6, 4+7) */
    float px = m[0]*1 + m[1]*1 + m[2]*1 + m[3];
    float py = m[4]*1 + m[5]*1 + m[6]*1 + m[7];
    float pz = m[8]*1 + m[9]*1 + m[10]*1 + m[11];
    CHECK_NEAR(px, 7, 1e-5); CHECK_NEAR(py, 9, 1e-5); CHECK_NEAR(pz, 11, 1e-5);
}

static void test_affine_mul_assoc_point(void) {
    /* A = rot z90 + translate; B = scale2 + translate; check (A∘B)·p == A·(B·p) */
    mrw_trs ta, tb;
    float c = (float)cos(M_PI / 4), s = (float)sin(M_PI / 4);
    float qa[4] = { 0, 0, s, c }; memcpy(ta.rot, qa, 16);
    ta.trans[0]=1; ta.trans[1]=2; ta.trans[2]=3; ta.scale[0]=ta.scale[1]=ta.scale[2]=1;
    float qi[4] = { 0,0,0,1 }; memcpy(tb.rot, qi, 16);
    tb.trans[0]=-1; tb.trans[1]=0; tb.trans[2]=4; tb.scale[0]=tb.scale[1]=tb.scale[2]=2;
    float A[12], B[12], AB[12];
    mrw_trs_to_affine(&ta, A);
    mrw_trs_to_affine(&tb, B);
    mrw_affine_mul(A, B, AB);
    float p[3] = { 0.5f, -1.5f, 2.0f };
    float Bp[3] = { B[0]*p[0]+B[1]*p[1]+B[2]*p[2]+B[3],
                    B[4]*p[0]+B[5]*p[1]+B[6]*p[2]+B[7],
                    B[8]*p[0]+B[9]*p[1]+B[10]*p[2]+B[11] };
    float ABp_seq[3] = { A[0]*Bp[0]+A[1]*Bp[1]+A[2]*Bp[2]+A[3],
                         A[4]*Bp[0]+A[5]*Bp[1]+A[6]*Bp[2]+A[7],
                         A[8]*Bp[0]+A[9]*Bp[1]+A[10]*Bp[2]+A[11] };
    float ABp[3] = { AB[0]*p[0]+AB[1]*p[1]+AB[2]*p[2]+AB[3],
                     AB[4]*p[0]+AB[5]*p[1]+AB[6]*p[2]+AB[7],
                     AB[8]*p[0]+AB[9]*p[1]+AB[10]*p[2]+AB[11] };
    for (int i = 0; i < 3; ++i) CHECK_NEAR(ABp[i], ABp_seq[i], 1e-4);
}

static void test_affine_mul_aliasing(void) {
    /* out aliasing an input must still be correct (impl uses a temp) */
    float A[12] = { 1,0,0,2, 0,1,0,3, 0,0,1,4 };
    float B[12] = { 0,-1,0,1, 1,0,0,0, 0,0,1,-2 };
    float ref[12]; mrw_affine_mul(A, B, ref);
    float t[12]; memcpy(t, A, sizeof t);
    mrw_affine_mul(t, B, t); /* out == a */
    for (int i = 0; i < 12; ++i) CHECK_NEAR(t[i], ref[i], 1e-6);
}

static void test_cofactor_similarity(void) {
    /* cof(s·R) == s²·R */
    float s = 3.0f, c = (float)cos(0.7), sn = (float)sin(0.7);
    float R[9] = { c,-sn,0, sn,c,0, 0,0,1 };
    float A[9]; for (int i = 0; i < 9; ++i) A[i] = s * R[i];
    float cof[9]; mrw_cofactor3(A, cof);
    for (int i = 0; i < 9; ++i) CHECK_NEAR(cof[i], s*s*R[i], 1e-4);
}

static void test_cofactor_invertible(void) {
    /* cof(A) == det(A)·(A^-1)^T for invertible A */
    float A[9] = { 2, 0.3f, -0.1f, 0.0f, 1.5f, 0.2f, 0.4f, -0.2f, 3.0f };
    double det = (double)A[0]*(A[4]*A[8]-A[5]*A[7]) - (double)A[1]*(A[3]*A[8]-A[5]*A[6]) + (double)A[2]*(A[3]*A[7]-A[4]*A[6]);
    float cof[9]; mrw_cofactor3(A, cof);
    /* (A^-1)^T = cof / det  ⇒  cof = det·(A^-1)^T; verify cof·A^T == det·I via cof * A^T */
    /* Simpler: verify A^T · cof == det·I  (since adj=cofᵀ, A·adj=det·I ⇒ A·cofᵀ=det·I) */
    float r[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[3*i+j] = A[3*i+0]*cof[3*j+0] + A[3*i+1]*cof[3*j+1] + A[3*i+2]*cof[3*j+2]; /* A · cofᵀ */
    CHECK_NEAR(r[0], det, 1e-3); CHECK_NEAR(r[4], det, 1e-3); CHECK_NEAR(r[8], det, 1e-3);
    CHECK_NEAR(r[1], 0, 1e-3); CHECK_NEAR(r[2], 0, 1e-3); CHECK_NEAR(r[3], 0, 1e-3);
    CHECK_NEAR(r[5], 0, 1e-3); CHECK_NEAR(r[6], 0, 1e-3); CHECK_NEAR(r[7], 0, 1e-3);
}

static void test_cofactor_singular(void) {
    /* zero-scale axis (singular) must still produce a defined cofactor (no inverse) */
    float A[9] = { 1,0,0, 0,1,0, 0,0,0 }; /* z-scale 0 */
    float cof[9]; mrw_cofactor3(A, cof);
    /* cof of diag(1,1,0) = diag(0,0,1) */
    float e[9] = { 0,0,0, 0,0,0, 0,0,1 };
    for (int i = 0; i < 9; ++i) CHECK_NEAR(cof[i], e[i], 1e-6);
}

static void test_half_decode(void) {
    CHECK_NEAR(mrw_half_to_float(0x3C00), 1.0f, 0);     /* 1.0 */
    CHECK_NEAR(mrw_half_to_float(0x0000), 0.0f, 0);     /* +0 */
    CHECK_NEAR(mrw_half_to_float(0xC000), -2.0f, 0);    /* -2.0 */
    CHECK_NEAR(mrw_half_to_float(0x3800), 0.5f, 0);     /* 0.5 */
    CHECK_NEAR(mrw_half_to_float(0x0001), (float)ldexp(1.0, -24), 1e-30); /* smallest subnormal */
    CHECK(mrw_half_to_float(0x7C00) == (float)INFINITY);
    CHECK(mrw_half_to_float(0xFC00) == -(float)INFINITY);
    CHECK(isnan(mrw_half_to_float(0x7E00)));
}

/* f32 → binary16 encoder (mrw_f32_to_f16): exact values, RNE tie behavior, classes, and the
 * full-range round-trip identity mrw_f32_to_f16(mrw_half_to_float(h)) == h for every NON-NaN h. */
static void test_f16_encode(void) {
    /* exact representable values + signed zero */
    CHECK(mrw_f32_to_f16(1.0f)   == 0x3C00u);
    CHECK(mrw_f32_to_f16(-2.0f)  == 0xC000u);
    CHECK(mrw_f32_to_f16(0.5f)   == 0x3800u);
    CHECK(mrw_f32_to_f16(0.0f)   == 0x0000u);
    CHECK(mrw_f32_to_f16(-0.0f)  == 0x8000u);
    /* overflow → ±inf; underflow → ±0 */
    CHECK(mrw_f32_to_f16(70000.0f)  == 0x7C00u);
    CHECK(mrw_f32_to_f16(-70000.0f) == 0xFC00u);
    CHECK(mrw_f32_to_f16(1e-9f)     == 0x0000u);
    /* smallest normal half (2^-14) and smallest subnormal half (2^-24) */
    CHECK(mrw_f32_to_f16((float)ldexp(1.0, -14)) == 0x0400u);
    CHECK(mrw_f32_to_f16((float)ldexp(1.0, -24)) == 0x0001u);
    /* NaN → a half NaN (exp all-ones, nonzero mantissa) */
    { uint16_t h = mrw_f32_to_f16((float)NAN); CHECK((h & 0x7C00u) == 0x7C00u && (h & 0x03FFu) != 0u); }

    /* round-to-nearest-EVEN ties. Step at the 1.0 binade is 2^-10; halfway is 2^-11.
     * 1.0 + 2^-11 is exactly halfway between 0x3C00 (even) and 0x3C01 (odd) → rounds DOWN to even. */
    CHECK(mrw_f32_to_f16(1.0f + (float)ldexp(1.0, -11)) == 0x3C00u);
    /* 1.0 + 3·2^-11 is halfway between 0x3C01 (odd) and 0x3C02 (even) → rounds UP to even. */
    CHECK(mrw_f32_to_f16(1.0f + (float)ldexp(3.0, -11)) == 0x3C02u);

    /* full-range identity: every finite/inf half decodes then re-encodes to itself. NaN payloads
     * are quieting-dependent (the encoder forces the quiet bit), so they are excluded. */
    for (uint32_t h = 0; h < 0x10000u; ++h) {
        int is_nan = ((h & 0x7C00u) == 0x7C00u) && ((h & 0x03FFu) != 0u);
        if (is_nan) continue;
        uint16_t r = mrw_f32_to_f16(mrw_half_to_float((uint16_t)h));
        if (r != (uint16_t)h) { printf("FAIL f16 round-trip h=0x%04X -> 0x%04X\n", h, r); ++g_fail; break; }
    }
}

int main(void) {
    test_quat_identity();
    test_quat_z90();
    test_trs_compose();
    test_affine_mul_assoc_point();
    test_affine_mul_aliasing();
    test_cofactor_similarity();
    test_cofactor_invertible();
    test_cofactor_singular();
    test_half_decode();
    test_f16_encode();
    printf(g_fail ? "test_math: %d FAILED\n" : "test_math: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
