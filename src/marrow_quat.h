/* marrow internal quaternion + vec3 kit - NOT part of the public ABI.
 *
 * One source of truth for the scalar quaternion/vector algebra shared by the pose-combine
 * primitives (marrow_blend.c), the IK solvers (marrow_ik.c), and root motion (marrow_pose.c).
 * Everything is `static inline`, so each TU compiles its own internal-linkage copy: zero added
 * exported ABI symbols.
 *
 * Quaternions are (x,y,z,w), Hamilton, column-vector convention. Rotations compose left-to-right
 * as q_total = q_outer ⊗ q_inner (apply inner first), matching mrw_affine_mul's A∘B "apply B
 * then A".
 *
 * NOT included by marrow_internal.h: the offline tools carry their own extern mrw_mat3_to_quat
 * (tools/authoring/mrw_decompose.c), so a single TU must never see both this static-inline copy
 * and that declaration. Only runtime src/ TUs include this header.
 */
#ifndef MRW_QUAT_H
#define MRW_QUAT_H

#include "marrow.h"   /* mrw_quat_to_mat3 */
#include <math.h>     /* sqrtf, sinf, cosf, fabsf */

/* ---- vec3 kit ---- */

static inline float mrw_v3_dot(const float a[3], const float b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static inline void mrw_v3_cross(const float a[3], const float b[3], float out[3]) {
    float x = a[1]*b[2] - a[2]*b[1];
    float y = a[2]*b[0] - a[0]*b[2];
    float z = a[0]*b[1] - a[1]*b[0];
    out[0] = x; out[1] = y; out[2] = z;        /* via temps: out may alias a or b */
}
static inline float mrw_v3_len(const float a[3]) { return sqrtf(mrw_v3_dot(a, a)); }
static inline void mrw_v3_add(const float a[3], const float b[3], float out[3]) {
    out[0] = a[0]+b[0]; out[1] = a[1]+b[1]; out[2] = a[2]+b[2];
}
static inline void mrw_v3_sub(const float a[3], const float b[3], float out[3]) {
    out[0] = a[0]-b[0]; out[1] = a[1]-b[1]; out[2] = a[2]-b[2];
}
static inline void mrw_v3_scale(const float a[3], float s, float out[3]) {
    out[0] = a[0]*s; out[1] = a[1]*s; out[2] = a[2]*s;
}
/* Normalize a into out; returns the original length. Degenerate (len ≤ ε) ⇒ out = (0,0,0),
 * returns the length so the caller can branch on a no-op fallback. */
static inline float mrw_v3_normalize(const float a[3], float out[3]) {
    float len = mrw_v3_len(a);
    if (len > 1e-12f) { float inv = 1.0f/len; out[0]=a[0]*inv; out[1]=a[1]*inv; out[2]=a[2]*inv; }
    else { out[0] = out[1] = out[2] = 0.0f; }
    return len;
}
/* Component of v perpendicular to a UNIT axis u: v − (v·u)·u. out may alias v. */
static inline void mrw_v3_reject(const float v[3], const float u[3], float out[3]) {
    float d = mrw_v3_dot(v, u);
    out[0] = v[0] - d*u[0]; out[1] = v[1] - d*u[1]; out[2] = v[2] - d*u[2];
}

/* ---- quaternion kit ---- */

/* Hamilton product o = a ⊗ b (apply b then a). o must not alias a or b. */
static inline void mrw_quat_mul(const float a[4], const float b[4], float o[4]) {
    float ax=a[0], ay=a[1], az=a[2], aw=a[3];
    float bx=b[0], by=b[1], bz=b[2], bw=b[3];
    o[0] = aw*bx + ax*bw + ay*bz - az*by;
    o[1] = aw*by - ax*bz + ay*bw + az*bx;
    o[2] = aw*bz + ax*by - ay*bx + az*bw;
    o[3] = aw*bw - ax*bx - ay*by - az*bz;
}
/* In-place normalize; degenerate (‖q‖² ≤ 0) ⇒ scale by 0 (matches the nlerp guard, which
 * yields the zero quaternion - itself the identity rotation through mrw_quat_to_mat3). */
static inline void mrw_quat_normalize(float q[4]) {
    float n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    float inv = (n2 > 0.0f) ? 1.0f/sqrtf(n2) : 0.0f;
    for (int k = 0; k < 4; ++k) q[k] *= inv;
}
/* Conjugate (= inverse for a unit quat): (−x,−y,−z,w). out may alias q. */
static inline void mrw_quat_conj(const float q[4], float out[4]) {
    out[0] = -q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = q[3];
}
/* Force q into the w ≥ 0 hemisphere (q ≡ −q as a rotation; the quaternion double cover). out may alias q. */
static inline void mrw_quat_canon(const float q[4], float out[4]) {
    float s = (q[3] < 0.0f) ? -1.0f : 1.0f;
    out[0] = s*q[0]; out[1] = s*q[1]; out[2] = s*q[2]; out[3] = s*q[3];
}
/* Rotate a vec3 by a quaternion via its 3×3 matrix. out must not alias v. */
static inline void mrw_quat_rotate(const float q[4], const float v[3], float out[3]) {
    float r[9]; mrw_quat_to_mat3(q, r);
    out[0] = r[0]*v[0] + r[1]*v[1] + r[2]*v[2];
    out[1] = r[3]*v[0] + r[4]*v[1] + r[5]*v[2];
    out[2] = r[6]*v[0] + r[7]*v[1] + r[8]*v[2];
}

/* 3×3 rotation (row-major) → unit quaternion (x,y,z,w) via Shepperd's method (pick the branch
 * with the largest denominator for stability). The exact inverse of mrw_quat_to_mat3; the
 * offline tools keep a separate extern copy (tools/authoring/mrw_decompose.c). */
static inline void mrw_mat3_to_quat(const float r[9], float q[4]) {
    float trace = r[0] + r[4] + r[8];
    float x, y, z, w;
    if (trace > 0.0f) {
        float S = sqrtf(trace + 1.0f) * 2.0f;              /* S = 4w */
        w = 0.25f * S; x = (r[7]-r[5])/S; y = (r[2]-r[6])/S; z = (r[3]-r[1])/S;
    } else if (r[0] > r[4] && r[0] > r[8]) {
        float S = sqrtf(1.0f + r[0] - r[4] - r[8]) * 2.0f; /* S = 4x */
        w = (r[7]-r[5])/S; x = 0.25f*S; y = (r[1]+r[3])/S; z = (r[2]+r[6])/S;
    } else if (r[4] > r[8]) {
        float S = sqrtf(1.0f + r[4] - r[0] - r[8]) * 2.0f; /* S = 4y */
        w = (r[2]-r[6])/S; x = (r[1]+r[3])/S; y = 0.25f*S; z = (r[5]+r[7])/S;
    } else {
        float S = sqrtf(1.0f + r[8] - r[0] - r[4]) * 2.0f; /* S = 4z */
        w = (r[3]-r[1])/S; x = (r[2]+r[6])/S; y = (r[5]+r[7])/S; z = 0.25f*S;
    }
    float n = sqrtf(x*x + y*y + z*z + w*w);
    float inv = (n > 0.0f) ? 1.0f/n : 0.0f;
    q[0] = x*inv; q[1] = y*inv; q[2] = z*inv; q[3] = w*inv;
}

/* Hemisphere-corrected slerp; near-parallel pairs fall back to normalized lerp (avoid 1/sinθ).
 * Used where true constant-angular-velocity is wanted (the IK weight blend itself uses nlerp;
 * slerp is available for callers that need it). out may not alias a or b0. */
static inline void mrw_quat_slerp(const float a[4], const float b0[4], float u, float out[4]) {
    float b[4]; float dot = a[0]*b0[0] + a[1]*b0[1] + a[2]*b0[2] + a[3]*b0[3];
    for (int k = 0; k < 4; ++k) b[k] = b0[k];
    if (dot < 0.0f) { for (int k = 0; k < 4; ++k) b[k] = -b[k]; dot = -dot; }
    if (dot > 0.999999f) {
        for (int k = 0; k < 4; ++k) out[k] = a[k] + u*(b[k]-a[k]);
        mrw_quat_normalize(out); return;
    }
    float theta = acosf(dot), st = sinf(theta);
    float wa = sinf((1.0f-u)*theta)/st, wb = sinf(u*theta)/st;
    for (int k = 0; k < 4; ++k) out[k] = wa*a[k] + wb*b[k];
    mrw_quat_normalize(out);
}

/* Shortest-arc rotation taking `from` onto `to`. Inputs need not be unit (normalized
 * internally). Aligned ⇒ identity; antiparallel ⇒ a π rotation about an arbitrary axis ⟂ from;
 * the general case uses the numerically-stable (cross, 1+cosθ) half-angle form. out distinct. */
static inline void mrw_quat_from_to(const float from[3], const float to[3], float out[4]) {
    float u[3], v[3];
    mrw_v3_normalize(from, u);
    mrw_v3_normalize(to, v);
    float d = mrw_v3_dot(u, v);
    if (d >= 1.0f - 1e-6f) { out[0]=out[1]=out[2]=0.0f; out[3]=1.0f; return; }   /* aligned */
    if (d <= -1.0f + 1e-6f) {                                                    /* antiparallel: π */
        float ax = fabsf(u[0]), ay = fabsf(u[1]), az = fabsf(u[2]);
        float t[3] = {0,0,0};
        if (ax <= ay && ax <= az) t[0] = 1.0f; else if (ay <= az) t[1] = 1.0f; else t[2] = 1.0f;
        float axis[3]; mrw_v3_cross(u, t, axis); mrw_v3_normalize(axis, axis);
        out[0]=axis[0]; out[1]=axis[1]; out[2]=axis[2]; out[3]=0.0f; return;
    }
    float c[3]; mrw_v3_cross(u, v, c);
    out[0]=c[0]; out[1]=c[1]; out[2]=c[2]; out[3]=1.0f + d;
    mrw_quat_normalize(out);
}

/* General axis-angle → unit quaternion. `axis` need not be unit (normalized internally);
 * a degenerate (near-zero) axis yields ≈ identity. */
static inline void mrw_quat_from_axis_angle(const float axis[3], float angle, float out[4]) {
    float a[3]; float len = mrw_v3_normalize(axis, a);
    float s = sinf(angle * 0.5f), c = cosf(angle * 0.5f);
    if (len > 1e-12f) { out[0]=a[0]*s; out[1]=a[1]*s; out[2]=a[2]*s; out[3]=c; }
    else { out[0]=out[1]=out[2]=0.0f; out[3]=1.0f; }
}

#endif /* MRW_QUAT_H */
