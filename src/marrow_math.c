/* marrow scalar math core - the scalar reference for every transform primitive.
 * The SIMD kernels are checked for parity against these implementations. */
#include "marrow.h"
#include <string.h>

/* unit quaternion q=(x,y,z,w) → 3×3 rotation, column-vector v'=R·v. Row-major. */
void mrw_quat_to_mat3(const float q[4], float out9[9]) {
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    out9[0] = 1.0f - 2.0f * (yy + zz); out9[1] = 2.0f * (xy - wz);        out9[2] = 2.0f * (xz + wy);
    out9[3] = 2.0f * (xy + wz);        out9[4] = 1.0f - 2.0f * (xx + zz); out9[5] = 2.0f * (yz - wx);
    out9[6] = 2.0f * (xz - wy);        out9[7] = 2.0f * (yz + wx);        out9[8] = 1.0f - 2.0f * (xx + yy);
}

/* compose TRS → 3×4 [A|t]; A[:,c] = R[:,c]·s_c (scale columns), t = translation. */
static void compose_affine(const float q[4], const float t[3], float sx, float sy, float sz, float out12[12]) {
    float r[9];
    mrw_quat_to_mat3(q, r);
    out12[0] = r[0] * sx; out12[1] = r[1] * sy; out12[2]  = r[2] * sz; out12[3]  = t[0];
    out12[4] = r[3] * sx; out12[5] = r[4] * sy; out12[6]  = r[5] * sz; out12[7]  = t[1];
    out12[8] = r[6] * sx; out12[9] = r[7] * sy; out12[10] = r[8] * sz; out12[11] = t[2];
}

void mrw_trs_to_affine(const mrw_trs *trs, float out_affine12[12]) {
    compose_affine(trs->rot, trs->trans, trs->scale[0], trs->scale[1], trs->scale[2], out_affine12);
}

void mrw_xform_to_affine(const mrw_xform *x, float out_affine12[12]) {
    compose_affine(x->rot, x->trans, x->scale, x->scale, x->scale, out_affine12);
}

/* compose two 3×4 affines C = A ∘ B (apply B then A). Output may alias inputs. */
void mrw_affine_mul(const float a12[12], const float b12[12], float out12[12]) {
    float c[12];
    for (int r = 0; r < 3; ++r) {
        const float a0 = a12[4 * r + 0], a1 = a12[4 * r + 1], a2 = a12[4 * r + 2], at = a12[4 * r + 3];
        /* C_A = A_A · B_A */
        c[4 * r + 0] = a0 * b12[0] + a1 * b12[4] + a2 * b12[8];
        c[4 * r + 1] = a0 * b12[1] + a1 * b12[5] + a2 * b12[9];
        c[4 * r + 2] = a0 * b12[2] + a1 * b12[6] + a2 * b12[10];
        /* C_t = A_A · B_t + A_t */
        c[4 * r + 3] = a0 * b12[3] + a1 * b12[7] + a2 * b12[11] + at;
    }
    memcpy(out12, c, sizeof c);
}

/* cofactor of a 3×3 (row-major): cof_{rc} = (−1)^{r+c}·minor_{rc}. Computed via
 * minors (NOT det·(A⁻¹)ᵀ) so it stays defined for singular A (e.g. zero-scale axis). */
void mrw_cofactor3(const float a[9], float out9[9]) {
    float c[9];
    c[0] =  (a[4] * a[8] - a[5] * a[7]);
    c[1] = -(a[3] * a[8] - a[5] * a[6]);
    c[2] =  (a[3] * a[7] - a[4] * a[6]);
    c[3] = -(a[1] * a[8] - a[2] * a[7]);
    c[4] =  (a[0] * a[8] - a[2] * a[6]);
    c[5] = -(a[0] * a[7] - a[1] * a[6]);
    c[6] =  (a[1] * a[5] - a[2] * a[4]);
    c[7] = -(a[0] * a[5] - a[2] * a[3]);
    c[8] =  (a[0] * a[4] - a[1] * a[3]);
    memcpy(out9, c, sizeof c);
}

void mrw_affine_cofactor(const float a12[12], float out9[9]) {
    const float a9[9] = { a12[0], a12[1], a12[2], a12[4], a12[5], a12[6], a12[8], a12[9], a12[10] };
    mrw_cofactor3(a9, out9);
}

/* IEEE-754 binary16 → binary32 (handles ±0, subnormals, inf/nan). */
float mrw_half_to_float(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) & 1u;
    uint32_t e = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)h & 0x3FFu;
    uint32_t out;
    if (e == 0) {
        if (m == 0) {
            out = s << 31;                                   /* ±0 */
        } else {
            e = 127 - 15 + 1;                                /* subnormal → normalized */
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            out = (s << 31) | (e << 23) | (m << 13);
        }
    } else if (e == 0x1Fu) {
        out = (s << 31) | 0x7F800000u | (m << 13);           /* inf / nan */
    } else {
        e = e - 15 + 127;
        out = (s << 31) | (e << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &out, sizeof f);
    return f;
}

/* IEEE-754 binary32 → binary16, round-to-nearest-even (handles subnormals, overflow→inf, NaN).
 * The exact-rounding partner of mrw_half_to_float, and the scalar/SSE2 fallback for the f16 output
 * palette (the AVX2 store uses F16C vcvtps2ph when present, which is bit-identical for finite inputs).
 * Byte-for-byte equal to the offline encoders (tools/authoring mrw_authoring_f32_to_half,
 * tools/bake bake_f32_to_half) - verified by the test_math bit-identity gate. */
uint16_t mrw_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t exp  = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp == 0xFF) {                                         /* inf/nan (NaN → quiet, payload kept) */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u | (mant >> 13) : 0));
    }
    int e = (int)exp - 127 + 15;
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);         /* overflow → inf */
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;                   /* underflow → ±0 */
        mant |= 0x800000u;                                    /* restore implicit 1 */
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t half_mant = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_mant & 1))) half_mant++;  /* RNE */
        return (uint16_t)(sign | half_mant);
    }
    uint32_t half_mant = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1))) {                 /* RNE */
        half_mant++;
        if (half_mant == 0x400u) { half_mant = 0; e++; if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | half_mant);
}
