/* See mrw_decompose.h. Matrix↔TRS decomposition for the offline authoring tools, reusing the
 * runtime scalar math (mrw_cofactor3 / mrw_xform_to_affine). The only genuinely new math is
 * mrw_mat3_to_quat - the inverse of mrw_quat_to_mat3. */
#include "mrw_decompose.h"

#include <math.h>
#include <string.h>

void mrw_mat3_to_quat(const float r[9], float q[4]) {
    /* Shepperd's method: pick the branch whose denominator is largest for stability. The
     * indices map the row-major rotation (r[0..8] = R00..R22) back to (x,y,z,w). */
    float trace = r[0] + r[4] + r[8];
    float x, y, z, w;
    if (trace > 0.0f) {
        float S = sqrtf(trace + 1.0f) * 2.0f;          /* S = 4w */
        w = 0.25f * S;
        x = (r[7] - r[5]) / S;
        y = (r[2] - r[6]) / S;
        z = (r[3] - r[1]) / S;
    } else if (r[0] > r[4] && r[0] > r[8]) {
        float S = sqrtf(1.0f + r[0] - r[4] - r[8]) * 2.0f; /* S = 4x */
        w = (r[7] - r[5]) / S;
        x = 0.25f * S;
        y = (r[1] + r[3]) / S;
        z = (r[2] + r[6]) / S;
    } else if (r[4] > r[8]) {
        float S = sqrtf(1.0f + r[4] - r[0] - r[8]) * 2.0f; /* S = 4y */
        w = (r[2] - r[6]) / S;
        x = (r[1] + r[3]) / S;
        y = 0.25f * S;
        z = (r[5] + r[7]) / S;
    } else {
        float S = sqrtf(1.0f + r[8] - r[0] - r[4]) * 2.0f; /* S = 4z */
        w = (r[3] - r[1]) / S;
        x = (r[2] + r[6]) / S;
        y = (r[5] + r[7]) / S;
        z = 0.25f * S;
    }
    float n = sqrtf(x*x + y*y + z*z + w*w);
    float inv = (n > 0.0f) ? 1.0f / n : 0.0f;
    q[0] = x*inv; q[1] = y*inv; q[2] = z*inv; q[3] = w*inv;
}

void mrw_affine_apply(const float m[12], const float p[3], float o[3]) {
    o[0] = m[0]*p[0] + m[1]*p[1] + m[2] *p[2] + m[3];
    o[1] = m[4]*p[0] + m[5]*p[1] + m[6] *p[2] + m[7];
    o[2] = m[8]*p[0] + m[9]*p[1] + m[10]*p[2] + m[11];
}

static void xform_identity(mrw_xform *o) {
    o->rot[0]=0; o->rot[1]=0; o->rot[2]=0; o->rot[3]=1;
    o->trans[0]=0; o->trans[1]=0; o->trans[2]=0; o->scale=1.0f;
}

/* det of a row-major 3×3 via its cofactors (row-0 expansion). */
static double mat3_det(const float a[9], const float cof[9]) {
    return (double)a[0]*cof[0] + (double)a[1]*cof[1] + (double)a[2]*cof[2];
}

int mrw_decompose_affine(const float m12[12], mrw_xform *out) {
    xform_identity(out); /* defined output even on the ineligible (return 0) path */

    float A[9] = { m12[0],m12[1],m12[2],  m12[4],m12[5],m12[6],  m12[8],m12[9],m12[10] };
    float t[3] = { m12[3], m12[7], m12[11] };
    for (int i = 0; i < 9; ++i) if (!isfinite(A[i])) return 0;
    for (int i = 0; i < 3; ++i) if (!isfinite(t[i])) return 0;

    float C0[9]; mrw_cofactor3(A, C0);
    double det = mat3_det(A, C0);
    if (!isfinite(det) || det <= 1e-8) return 0;   /* near-singular (≤1e-8) or reflection (<0) */

    /* polar iteration Rₖ₊₁ = ½(Rₖ + Rₖ⁻ᵀ), Rₖ⁻ᵀ = cof(Rₖ)/det(Rₖ) */
    float R[9]; memcpy(R, A, sizeof R);
    int converged = 0;
    for (int it = 0; it < 32; ++it) {
        float C[9]; mrw_cofactor3(R, C);
        double d = mat3_det(R, C);
        if (!isfinite(d) || fabs(d) < 1e-12) return 0;
        float diff = 0.0f, Rn[9];
        for (int i = 0; i < 9; ++i) {
            Rn[i] = 0.5f * (R[i] + (float)((double)C[i] / d));
            float e = fabsf(Rn[i] - R[i]); if (e > diff) diff = e;
        }
        memcpy(R, Rn, sizeof R);
        if (diff < 1e-7f) { converged = 1; break; }
    }
    if (!converged) return 0;

    /* R must be proper-orthonormal: Rᵀ·R ≈ I and det(R) ≈ +1 */
    for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
            double dij = (double)R[i*3]*R[j*3] + (double)R[i*3+1]*R[j*3+1] + (double)R[i*3+2]*R[j*3+2];
            double target = (i == j) ? 1.0 : 0.0;
            if (fabs(dij - target) > 1e-3) return 0;
        }
    }
    float CR[9]; mrw_cofactor3(R, CR);
    double detR = mat3_det(R, CR);
    if (!(detR > 0.9 && detR < 1.1)) return 0;

    /* uniform scale s = trace(Rᵀ·A)/3 = ⟨R,A⟩_F / 3 (mean singular value) */
    double fro = 0.0; for (int i = 0; i < 9; ++i) fro += (double)R[i]*A[i];
    float s = (float)(fro / 3.0);
    if (!isfinite(s) || !(s > 0.0f)) return 0;

    mrw_mat3_to_quat(R, out->rot);
    out->trans[0]=t[0]; out->trans[1]=t[1]; out->trans[2]=t[2];
    out->scale = s;
    return 1;
}

float mrw_affine_probe_dist(const float a[12], const float b[12], uint32_t np, const float *probes) {
    float worst = 0.0f;
    for (uint32_t i = 0; i < np; ++i) {
        float pa[3], pb[3];
        mrw_affine_apply(a, probes + (size_t)i*3, pa);
        mrw_affine_apply(b, probes + (size_t)i*3, pb);
        float dx = pa[0]-pb[0], dy = pa[1]-pb[1], dz = pa[2]-pb[2];
        float d = sqrtf(dx*dx + dy*dy + dz*dz);
        if (d > worst) worst = d;
    }
    return worst;
}

float mrw_decompose_residual(const float m12[12], uint32_t np, const float *probes) {
    mrw_xform x;
    if (!mrw_decompose_affine(m12, &x)) return INFINITY;
    float mp[12]; mrw_xform_to_affine(&x, mp);
    return mrw_affine_probe_dist(m12, mp, np, probes);
}
