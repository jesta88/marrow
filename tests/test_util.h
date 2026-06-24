#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "marrow.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int g_fail = 0;

#define CHECK(cond) do { if (!(cond)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } } while (0)

#define CHECK_EQ(a, b) do { long long _a = (long long)(a), _b = (long long)(b); \
    if (_a != _b) { printf("FAIL %s:%d: %s (%lld) != %s (%lld)\n", __FILE__, __LINE__, #a, _a, #b, _b); ++g_fail; } } while (0)

#define CHECK_NEAR(a, b, eps) do { double _d = (double)(a) - (double)(b); if (_d < 0) _d = -_d; \
    if (_d > (eps)) { printf("FAIL %s:%d: |%s - %s| = %g > %g\n", __FILE__, __LINE__, #a, #b, _d, (double)(eps)); ++g_fail; } } while (0)

#define TEST_MAIN_RETURN() return (g_fail == 0) ? 0 : 1

/* compare two 3x4 affines (12 floats) */
static int aff_near(const float a[12], const float b[12], double eps) {
    for (int i = 0; i < 12; ++i) { double d = (double)a[i] - b[i]; if (d < 0) d = -d; if (d > eps) return 0; }
    return 1;
}

/* invert a 3x4 affine [A|t] -> [A^-1 | -A^-1 t]; returns 0 if singular */
static int aff_inverse(const float m[12], float out[12]) {
    float a[9] = { m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10] };
    double det = (double)a[0] * (a[4]*a[8] - a[5]*a[7])
               - (double)a[1] * (a[3]*a[8] - a[5]*a[6])
               + (double)a[2] * (a[3]*a[7] - a[4]*a[6]);
    if (fabs(det) < 1e-20) return 0;
    float c[9];
    mrw_cofactor3(a, c);                 /* cofactor matrix C_rc */
    float ai[9];
    for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 3; ++col)
            ai[3*r + col] = (float)(c[3*col + r] / det);  /* A^-1 = C^T / det */
    float t[3] = { m[3], m[7], m[11] };
    out[0]=ai[0]; out[1]=ai[1]; out[2]=ai[2]; out[3]=-(ai[0]*t[0]+ai[1]*t[1]+ai[2]*t[2]);
    out[4]=ai[3]; out[5]=ai[4]; out[6]=ai[5]; out[7]=-(ai[3]*t[0]+ai[4]*t[1]+ai[5]*t[2]);
    out[8]=ai[6]; out[9]=ai[7]; out[10]=ai[8]; out[11]=-(ai[6]*t[0]+ai[7]*t[1]+ai[8]*t[2]);
    return 1;
}

static const float MRW_IDENTITY12[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };

#define MRW_TEST_MAXJOINTS 256

/* Compute inverse_bind[j] = inverse(bind_model[j]) where bind_model is the rest pose
 * run through the hierarchy. rest_local is jc*10 (q,t,s3); out_ib is jc*12. */
static void compute_bind_inverse(uint32_t jc, const uint16_t *parent, const float *rest_local, float *out_ib) {
    static float model[MRW_TEST_MAXJOINTS * 12];
    for (uint32_t j = 0; j < jc; ++j) {
        mrw_trs trs;
        memcpy(trs.rot,   rest_local + j*10 + 0, 16);
        memcpy(trs.trans, rest_local + j*10 + 4, 12);
        memcpy(trs.scale, rest_local + j*10 + 7, 12);
        float local12[12];
        mrw_trs_to_affine(&trs, local12);
        if (parent[j] == 0xFFFF) memcpy(model + j*12, local12, sizeof local12);
        else mrw_affine_mul(model + (size_t)parent[j]*12, local12, model + j*12);
        aff_inverse(model + j*12, out_ib + j*12);
    }
}

#endif /* TEST_UTIL_H */
