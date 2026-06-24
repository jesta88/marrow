/* matrix↔TRS decomposition for the offline authoring tools (gltf2marrow, marrow-bake).
 * The inverse of the runtime's compose math: a rotation matrix → quaternion (Shepperd) and a
 * 3×4 affine → similarity (q, t, uniform scale) via polar decomposition. It reuses the
 * runtime scalar math (mrw_cofactor3 / mrw_xform_to_affine); the only genuinely new math is the
 * inverse of mrw_quat_to_mat3. Lives in tools/ and is never linked into the zero-dep runtime. */
#ifndef MRW_DECOMPOSE_H
#define MRW_DECOMPOSE_H

#include "marrow.h"

/* mat3 (row-major, column-vector v'=R·v) → unit quaternion (x,y,z,w). The inverse of
 * mrw_quat_to_mat3, via Shepperd's method (trace / largest-diagonal branch). The
 * sign of q is arbitrary (q and −q encode the same rotation); the result is renormalized. */
void mrw_mat3_to_quat(const float r9[9], float out_q[4]);

/* Polar-decompose a 3×4 affine M=[A|t] into a similarity (q, t, uniform s).
 * Iteration Rₖ₊₁ = ½(Rₖ + Rₖ⁻ᵀ) with Rₖ⁻ᵀ = cof(Rₖ)/det(Rₖ); s = trace(Rᵀ·A)/3.
 * Returns 1 and fills *out (q renormalized, trans=t, scale=s) when the decomposition is
 * STRUCTURALLY well-formed; returns 0 (bone ineligible) when A is non-finite, det(A) ≤ 1e-8
 * (near-singular) or < 0 (reflection), the iteration fails to converge, R is not
 * proper-orthonormal, or s is non-finite/non-positive. (The residual ≤ tol test is applied
 * on top of this by the caller.) */
int mrw_decompose_affine(const float m12[12], mrw_xform *out);

/* p' = M·[p,1] for a 3×4 affine. */
void mrw_affine_apply(const float m12[12], const float p[3], float out[3]);

/* Reconstruction residual: reconstruct M' = compose(q,t,s) from a decomposition of M and return
 * max over the `np` bind-space probe points (np·3 floats) of ‖M·[p,1] − M'·[p,1]‖ (meters).
 * If M is structurally ineligible, returns +INFINITY (so it always exceeds tol). */
float mrw_decompose_residual(const float m12[12], uint32_t np, const float *probes);

/* Worst probe displacement between two 3×4 affines: max over `np` bind-space probe points of
 * ‖A·[p,1] − B·[p,1]‖ (meters). The CPU-vs-baked "pop" / cross-fade-gap metric. */
float mrw_affine_probe_dist(const float a12[12], const float b12[12], uint32_t np, const float *probes);

#endif /* MRW_DECOMPOSE_H */
