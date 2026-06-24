/* marrow inverse kinematics - analytic single-effector solvers: two-bone (law-of-cosines) and
 * aim/look-at. These adjust LOCAL rotations in `locals` to meet a model-space goal; the caller
 * re-runs mrw_local_to_model over the affected subtree afterward. IK is CPU only (never batched,
 * never on the GPU). Scalar; reuses the shared quat/vec kit (marrow_quat.h). */
#include "marrow_internal.h"   /* mrw_clampf, mrw_quat_nlerp, mrw_f32_finite */
#include "marrow_quat.h"
#include <math.h>              /* acosf, sinf, atan2f, fabsf */

/* ------------------------------------------------------------------ spaces / scale-strip */

/* Joint model-space position = the model 3×4 translation column. */
static void ik_pos(const float *model, uint32_t j, float out[3]) {
    const float *m = model + (size_t)j * 12;
    out[0] = m[3]; out[1] = m[7]; out[2] = m[11];
}

static int vec3_finite(const float v[3]) {
    return mrw_f32_finite(v[0]) && mrw_f32_finite(v[1]) && mrw_f32_finite(v[2]);
}
static int quat_finite(const float q[4]) {
    return mrw_f32_finite(q[0]) && mrw_f32_finite(q[1]) && mrw_f32_finite(q[2]) && mrw_f32_finite(q[3]);
}
static int model_entry_finite(const float *model, uint32_t j) {
    const float *m = model + (size_t)j * 12;
    for (int k = 0; k < 12; ++k) if (!mrw_f32_finite(m[k])) return 0;
    return 1;
}

/* Derive a joint's WORLD rotation by stripping UNIFORM scale from its model 3×3: each basis
 * column normalized by its length. Returns 1 + the quaternion on success; 0 (⇒ MRW_E_UNSUPPORTED)
 * when the 3×3 is not a proper, uniformly-scaled rotation - any column length ≤ ε (degenerate /
 * zero scale), lengths unequal (non-uniform), normalized columns non-orthonormal (shear), or
 * det ≤ 0 (reflection / negative uniform scale such as A = −R, which passes the length/orthogonality
 * tests but is not a rotation). The chain AND its ancestors MUST satisfy this. */
static int ik_world_rot(const float *model, uint32_t j, float out_q[4]) {
    const float *m = model + (size_t)j * 12;
    float col[3][3], len[3];
    for (int c = 0; c < 3; ++c) {
        col[c][0] = m[0*4+c]; col[c][1] = m[1*4+c]; col[c][2] = m[2*4+c];
        len[c] = sqrtf(col[c][0]*col[c][0] + col[c][1]*col[c][1] + col[c][2]*col[c][2]);
        if (!(len[c] > 1e-6f)) return 0;                    /* zero-length column */
    }
    float lmax = len[0]; if (len[1] > lmax) lmax = len[1]; if (len[2] > lmax) lmax = len[2];
    for (int c = 0; c < 3; ++c)
        for (int d = c + 1; d < 3; ++d)
            if (fabsf(len[c] - len[d]) > 1e-3f * lmax) return 0; /* non-uniform scale */

    float R[9];
    for (int c = 0; c < 3; ++c) {
        float inv = 1.0f / len[c];
        R[0*3+c] = col[c][0]*inv; R[1*3+c] = col[c][1]*inv; R[2*3+c] = col[c][2]*inv;
    }
    for (int c = 0; c < 3; ++c)                              /* shear: columns must be orthogonal */
        for (int d = c + 1; d < 3; ++d) {
            float dot = R[0*3+c]*R[0*3+d] + R[1*3+c]*R[1*3+d] + R[2*3+c]*R[2*3+d];
            if (fabsf(dot) > 1e-3f) return 0;
        }
    float det = R[0]*(R[4]*R[8] - R[5]*R[7])
              - R[1]*(R[3]*R[8] - R[5]*R[6])
              + R[2]*(R[3]*R[7] - R[4]*R[6]);
    if (!(det > 0.0f)) return 0;                            /* reflection / negative uniform scale */
    mrw_mat3_to_quat(R, out_q);
    return 1;
}

/* ------------------------------------------------------------------ two-bone */

/* Bend direction (unit, ⟂ û, toward the pole): normalized rejection of `pref` (= P−A) from û;
 * fall back to the rejection of `alt` (= B−A, the current bend) when the pole is on the bone axis,
 * then to any axis ⟂ û when the chain is straight. û is unit. */
static int try_perp(const float v[3], const float u[3], float out[3]) {
    float r[3]; mrw_v3_reject(v, u, r);
    float rl = mrw_v3_len(r), vl = mrw_v3_len(v);
    float thresh = 1e-6f * (vl > 1.0f ? vl : 1.0f);
    if (rl > thresh) { mrw_v3_scale(r, 1.0f / rl, out); return 1; }
    return 0;
}
static void bend_dir(const float pref[3], const float alt[3], const float u[3], float out[3]) {
    if (try_perp(pref, u, out)) return;
    if (try_perp(alt,  u, out)) return;
    float ax = fabsf(u[0]), ay = fabsf(u[1]), az = fabsf(u[2]);
    float t[3] = {0,0,0};
    if (ax <= ay && ax <= az) t[0] = 1.0f; else if (ay <= az) t[1] = 1.0f; else t[2] = 1.0f;
    float r[3]; mrw_v3_cross(u, t, r); mrw_v3_normalize(r, out);
}

mrw_result mrw_ik_two_bone(const mrw_skeleton_view *skel, const float *model, mrw_trs *locals,
                           uint32_t root_j, uint32_t mid_j, uint32_t end_j,
                           const float target_model[3], const float pole_model[3],
                           float weight, uint32_t joint_capacity) {
    if (!skel || !model || !locals || !target_model || !pole_model) return MRW_E_RANGE;
    uint32_t n = skel->joint_count;
    if (n > joint_capacity) return MRW_E_CAPACITY;
    if (root_j >= n || mid_j >= n || end_j >= n) return MRW_E_RANGE;
    if (!mrw_f32_finite(weight)) return MRW_E_RANGE;
    if (!vec3_finite(target_model) || !vec3_finite(pole_model)) return MRW_E_RANGE;

    /* chain MUST be parent-linked */
    uint16_t par_mid = 0, par_end = 0, par_root = 0;
    mrw_skeleton_parent(skel, mid_j, &par_mid);
    mrw_skeleton_parent(skel, end_j, &par_end);
    mrw_skeleton_parent(skel, root_j, &par_root);
    if (par_mid != root_j || par_end != mid_j) return MRW_E_INCOMPATIBLE;

    /* finite-validate every model/local entry IK consumes - no partial write on failure */
    if (!model_entry_finite(model, root_j) || !model_entry_finite(model, mid_j) ||
        !model_entry_finite(model, end_j)) return MRW_E_RANGE;
    if (par_root != 0xFFFF && !model_entry_finite(model, par_root)) return MRW_E_RANGE;
    if (!quat_finite(locals[root_j].rot) || !quat_finite(locals[mid_j].rot)) return MRW_E_RANGE;

    float wc = mrw_clampf(weight, 0.0f, 1.0f);
    if (wc == 0.0f) return MRW_OK;                          /* weight 0 ⇒ exact no-op */

    /* world rotations (uniform-scale-stripped); non-similarity ⇒ MRW_E_UNSUPPORTED. The whole
     * chain AND its ancestors MUST be a similarity: root/mid/parent supply world rotations; the
     * solve uses only end_j's POSITION, but end_j is part of the chain, so its similarity is enforced
     * too (a non-uniform / sheared / reflected end is rejected before any write). */
    float Rroot[4], Rmid[4], Rend[4], Rparent[4] = {0,0,0,1};
    if (!ik_world_rot(model, root_j, Rroot)) return MRW_E_UNSUPPORTED;
    if (!ik_world_rot(model, mid_j,  Rmid))  return MRW_E_UNSUPPORTED;
    if (!ik_world_rot(model, end_j,  Rend))  return MRW_E_UNSUPPORTED;   /* chain-similarity check */
    if (par_root != 0xFFFF && !ik_world_rot(model, par_root, Rparent)) return MRW_E_UNSUPPORTED;
    (void)Rend;                                            /* checked for similarity; rotation unused */

    /* positions + bone lengths from the model translation columns */
    float A[3], B[3], C[3];
    ik_pos(model, root_j, A); ik_pos(model, mid_j, B); ik_pos(model, end_j, C);
    float ba[3], cb[3];
    mrw_v3_sub(B, A, ba); float l1 = mrw_v3_len(ba);
    mrw_v3_sub(C, B, cb); float l2 = mrw_v3_len(cb);
    if (!(l1 > 1e-6f) || !(l2 > 1e-6f)) return MRW_OK;      /* degenerate bone ⇒ no-op */

    /* reach + direction to target */
    float ta[3]; mrw_v3_sub(target_model, A, ta);
    float reach = mrw_v3_len(ta);
    float u[3];
    if (mrw_v3_normalize(ta, u) <= 1e-6f) return MRW_OK;    /* target ≈ root ⇒ no-op */
    float d = mrw_clampf(reach, fabsf(l1 - l2), l1 + l2);

    /* law of cosines: root angle α between û and the upper bone */
    float ca = mrw_clampf((l1*l1 + d*d - l2*l2) / (2.0f*l1*d), -1.0f, 1.0f);
    float alpha = acosf(ca), sa = sinf(alpha);

    /* bend toward the pole, then the solved mid/end positions */
    float pa[3]; mrw_v3_sub(pole_model, A, pa);
    float bhat[3]; bend_dir(pa, ba, u, bhat);
    float Bp[3], Cp[3];
    for (int k = 0; k < 3; ++k) Bp[k] = A[k] + l1 * (ca*u[k] + sa*bhat[k]);
    for (int k = 0; k < 3; ++k) Cp[k] = A[k] + d * u[k];

    /* bone swings (swing only; input twist preserved): root then mid in the root-solved frame */
    float bpa[3]; mrw_v3_sub(Bp, A, bpa);
    float q1[4]; mrw_quat_from_to(ba, bpa, q1);
    float Rroot_new[4]; mrw_quat_mul(q1, Rroot, Rroot_new); mrw_quat_normalize(Rroot_new);

    float q1cb[3]; mrw_quat_rotate(q1, cb, q1cb);           /* q1·(C−B) */
    float cpbp[3]; mrw_v3_sub(Cp, Bp, cpbp);
    float q2[4]; mrw_quat_from_to(q1cb, cpbp, q2);
    float q1Rmid[4]; mrw_quat_mul(q1, Rmid, q1Rmid);
    float Rmid_new[4]; mrw_quat_mul(q2, q1Rmid, Rmid_new); mrw_quat_normalize(Rmid_new);

    /* write-back: local = conj(parent_world) ⊗ world_new, blended with the input at weight
     * (nlerp). The mid uses the SOLVED root world rotation as its parent. */
    float croot[4]; mrw_quat_conj(Rparent, croot);
    float lroot[4]; mrw_quat_mul(croot, Rroot_new, lroot); mrw_quat_normalize(lroot);
    float cmid[4]; mrw_quat_conj(Rroot_new, cmid);
    float lmid[4]; mrw_quat_mul(cmid, Rmid_new, lmid); mrw_quat_normalize(lmid);

    float out_root[4], out_mid[4];
    mrw_quat_nlerp(locals[root_j].rot, lroot, wc, out_root);
    mrw_quat_nlerp(locals[mid_j].rot,  lmid,  wc, out_mid);
    for (int k = 0; k < 4; ++k) locals[root_j].rot[k] = out_root[k];
    for (int k = 0; k < 4; ++k) locals[mid_j].rot[k]  = out_mid[k];
    return MRW_OK;
}

/* ------------------------------------------------------------------ aim (look-at) */

mrw_result mrw_ik_aim(const mrw_skeleton_view *skel, const float *model, mrw_trs *locals,
                      uint32_t joint, const float aim_axis_local[3], const float up_axis_local[3],
                      const float target_model[3], const float up_model[3],
                      float weight, uint32_t joint_capacity) {
    if (!skel || !model || !locals || !aim_axis_local || !up_axis_local || !target_model || !up_model)
        return MRW_E_RANGE;
    uint32_t n = skel->joint_count;
    if (n > joint_capacity) return MRW_E_CAPACITY;
    if (joint >= n) return MRW_E_RANGE;
    if (!mrw_f32_finite(weight)) return MRW_E_RANGE;
    if (!vec3_finite(aim_axis_local) || !vec3_finite(up_axis_local) ||
        !vec3_finite(target_model) || !vec3_finite(up_model)) return MRW_E_RANGE;

    /* aim/up local axes MUST be non-collinear (the up reference fixes twist) */
    float an[3], un[3];
    if (mrw_v3_normalize(aim_axis_local, an) <= 1e-6f) return MRW_E_RANGE;
    if (mrw_v3_normalize(up_axis_local,  un) <= 1e-6f) return MRW_E_RANGE;
    { float cr[3]; mrw_v3_cross(an, un, cr); if (mrw_v3_len(cr) <= 1e-4f) return MRW_E_RANGE; }

    uint16_t par = 0; mrw_skeleton_parent(skel, joint, &par);
    if (!model_entry_finite(model, joint)) return MRW_E_RANGE;
    if (par != 0xFFFF && !model_entry_finite(model, par)) return MRW_E_RANGE;
    if (!quat_finite(locals[joint].rot)) return MRW_E_RANGE;

    float wc = mrw_clampf(weight, 0.0f, 1.0f);
    if (wc == 0.0f) return MRW_OK;                          /* weight 0 ⇒ exact no-op */

    float Rj[4], Rparent[4] = {0,0,0,1};
    if (!ik_world_rot(model, joint, Rj)) return MRW_E_UNSUPPORTED;
    if (par != 0xFFFF && !ik_world_rot(model, par, Rparent)) return MRW_E_UNSUPPORTED;

    float Pj[3]; ik_pos(model, joint, Pj);
    float fvec[3]; mrw_v3_sub(target_model, Pj, fvec);
    float fhat[3];
    if (mrw_v3_normalize(fvec, fhat) <= 1e-6f) return MRW_OK; /* target ≈ joint ⇒ no-op */

    /* swing the aim axis onto the target direction */
    float aim_world[3]; mrw_quat_rotate(Rj, aim_axis_local, aim_world);
    float q1[4]; mrw_quat_from_to(aim_world, fhat, q1);
    float R1[4]; mrw_quat_mul(q1, Rj, R1); mrw_quat_normalize(R1);

    /* roll to up (signed). Skip if up_model ∥ f̂ or the current up projects to ≈0 on the plane ⟂ f̂. */
    float Rp[4];
    float that[3], chat[3];
    int roll = 1;
    { float upproj[3]; mrw_v3_reject(up_model, fhat, upproj);
      if (mrw_v3_normalize(upproj, that) <= 1e-6f) roll = 0; }
    if (roll) {
        float upw[3]; mrw_quat_rotate(R1, up_axis_local, upw);
        float cproj[3]; mrw_v3_reject(upw, fhat, cproj);
        if (mrw_v3_normalize(cproj, chat) <= 1e-6f) roll = 0;
    }
    if (roll) {
        float cxt[3]; mrw_v3_cross(chat, that, cxt);
        float rho = atan2f(mrw_v3_dot(fhat, cxt), mrw_v3_dot(chat, that));
        float qroll[4]; mrw_quat_from_axis_angle(fhat, rho, qroll);
        mrw_quat_mul(qroll, R1, Rp); mrw_quat_normalize(Rp);
    } else {
        for (int k = 0; k < 4; ++k) Rp[k] = R1[k];
    }

    /* write-back through the (scale-stripped) parent world rotation, blended at weight */
    float cpar[4]; mrw_quat_conj(Rparent, cpar);
    float lsolved[4]; mrw_quat_mul(cpar, Rp, lsolved); mrw_quat_normalize(lsolved);
    float outq[4]; mrw_quat_nlerp(locals[joint].rot, lsolved, wc, outq);
    for (int k = 0; k < 4; ++k) locals[joint].rot[k] = outq[k];
    return MRW_OK;
}
