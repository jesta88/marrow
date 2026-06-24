/* marrow pose algebra - component-space blend / additive (make + accumulate), the CPU
 * blend/additive/mask layer. Pure, scalar, zero-allocation functions over local-pose
 * (mrw_trs) arrays, slotting between mrw_clip_sample_local and mrw_local_to_model. These
 * single-pose forms are the reference the batch/SIMD kernels are checked against bit-for-bit. */
#include "marrow_internal.h"   /* mrw_clampf, mrw_quat_nlerp, mrw_interp_trs, mrw_f32_finite */
#include "marrow_quat.h"       /* mrw_quat_mul / _conj / _canon / _normalize, mrw_v3_sub */

/* ------------------------------------------------------------------ shared validation */

/* Every input array is finite-validated up front so a failure writes NO output. A pose
 * component, mask value, or weight that is inf/nan ⇒ MRW_E_RANGE. */
static int trs_finite(const mrw_trs *t) {
    for (int k = 0; k < 4; ++k) if (!mrw_f32_finite(t->rot[k]))   return 0;
    for (int k = 0; k < 3; ++k) if (!mrw_f32_finite(t->trans[k])) return 0;
    for (int k = 0; k < 3; ++k) if (!mrw_f32_finite(t->scale[k])) return 0;
    return 1;
}
static int trs_array_finite(const mrw_trs *a, uint32_t n) {
    for (uint32_t j = 0; j < n; ++j) if (!trs_finite(&a[j])) return 0;
    return 1;
}
static int floats_finite(const float *a, uint32_t n) {
    for (uint32_t j = 0; j < n; ++j) if (!mrw_f32_finite(a[j])) return 0;
    return 1;
}

/* Effective per-joint weight: clamp w and mask[j] to [0,1] individually, then their product.
 * mask NULL ⇒ uniform w. wc is the already-clamped global weight. */
static inline float eff_weight(float wc, const float *mask, uint32_t j) {
    float mj = mask ? mrw_clampf(mask[j], 0.0f, 1.0f) : 1.0f;
    return mrw_clampf(wc * mj, 0.0f, 1.0f);
}

/* ------------------------------------------------------------------ blend */

mrw_result mrw_pose_blend(const mrw_trs *a, const mrw_trs *b, float w,
                          const float *mask, mrw_trs *out,
                          uint32_t joint_count, uint32_t joint_capacity) {
    if (joint_count == 0) return MRW_OK;                /* no-op (data pointers may be NULL) */
    if (!a || !b || !out) return MRW_E_RANGE;
    if (joint_count > joint_capacity) return MRW_E_CAPACITY;
    if (!mrw_f32_finite(w)) return MRW_E_RANGE;
    if (!trs_array_finite(a, joint_count)) return MRW_E_RANGE;
    if (!trs_array_finite(b, joint_count)) return MRW_E_RANGE;
    if (mask && !floats_finite(mask, joint_count)) return MRW_E_RANGE;

    float wc = mrw_clampf(w, 0.0f, 1.0f);
    for (uint32_t j = 0; j < joint_count; ++j) {
        float we = eff_weight(wc, mask, j);
        /* nlerp the rotation (hemisphere-corrected), lerp trans/scale - exactly mrw_interp_trs.
         * we=0 ⇒ trans/scale are a[j] exactly (mul-by-zero) and rot = normalize(a[j].rot) ≈ a[j]
         * as a rotation; we=1 ⇒ trans/scale = b[j], rot = b[j].rot up to sign (q ≡ −q). out[j] MAY
         * exactly alias a[j] or b[j] - mrw_interp_trs reads both endpoints before writing. */
        mrw_interp_trs(&a[j], &b[j], we, &out[j]);
    }
    return MRW_OK;
}

/* ------------------------------------------------------------------ make additive */

mrw_result mrw_pose_make_additive(const mrw_trs *pose, const mrw_trs *base,
                                  mrw_trs *out_delta, uint32_t joint_count, uint32_t joint_capacity) {
    if (joint_count == 0) return MRW_OK;                /* no-op (data pointers may be NULL) */
    if (!pose || !base || !out_delta) return MRW_E_RANGE;
    if (joint_count > joint_capacity) return MRW_E_CAPACITY;
    if (!trs_array_finite(pose, joint_count)) return MRW_E_RANGE;
    if (!trs_array_finite(base, joint_count)) return MRW_E_RANGE;

    for (uint32_t j = 0; j < joint_count; ++j) {
        /* rotation delta = canon(normalize(conj(base) ⊗ pose)): canon ⇒ w≥0, so a same-rotation /
         * opposite-sign base/pose pair maps to +identity, not the −identity that would spin 180° at
         * partial accumulate weight. */
        float cb[4]; mrw_quat_conj(base[j].rot, cb);
        float dr[4]; mrw_quat_mul(cb, pose[j].rot, dr);
        mrw_quat_normalize(dr);
        mrw_quat_canon(dr, dr);
        /* translation delta = pose − base (identity 0) */
        float dt[3]; mrw_v3_sub(pose[j].trans, base[j].trans, dt);
        /* scale delta = pose ⊘ base RATIO (identity 1); a non-positive base, or a non-finite /
         * non-positive ratio, maps to 1 (no scale delta - a documented round-trip exception). */
        float ds[3];
        for (int k = 0; k < 3; ++k) {
            float bs = base[j].scale[k];
            float r = pose[j].scale[k] / bs;
            ds[k] = (bs > 0.0f && mrw_f32_finite(r) && r > 0.0f) ? r : 1.0f;
        }
        /* write last - out_delta[j] MAY exactly alias pose[j] or base[j] (all reads are above). */
        for (int k = 0; k < 4; ++k) out_delta[j].rot[k]   = dr[k];
        for (int k = 0; k < 3; ++k) out_delta[j].trans[k] = dt[k];
        for (int k = 0; k < 3; ++k) out_delta[j].scale[k] = ds[k];
    }
    return MRW_OK;
}

/* ------------------------------------------------------------------ accumulate */

static const float MRW_QUAT_IDENTITY[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

mrw_result mrw_pose_accumulate(const mrw_trs *base, const mrw_trs *delta, float w,
                               const float *mask, mrw_trs *out,
                               uint32_t joint_count, uint32_t joint_capacity) {
    if (joint_count == 0) return MRW_OK;                /* no-op (data pointers may be NULL) */
    if (!base || !delta || !out) return MRW_E_RANGE;
    if (joint_count > joint_capacity) return MRW_E_CAPACITY;
    if (!mrw_f32_finite(w)) return MRW_E_RANGE;
    if (!trs_array_finite(base,  joint_count)) return MRW_E_RANGE;
    if (!trs_array_finite(delta, joint_count)) return MRW_E_RANGE;
    if (mask && !floats_finite(mask, joint_count)) return MRW_E_RANGE;

    float wc = mrw_clampf(w, 0.0f, 1.0f);
    for (uint32_t j = 0; j < joint_count; ++j) {
        float we = eff_weight(wc, mask, j);
        /* nlerp_id(delta.rot, we) = normalize(lerp(identity, delta.rot, we)) - the nlerp with a
         * fixed identity endpoint, NO trig (so the batch kernels vectorize). delta.rot canonical
         * (w≥0) ⇒ the hemisphere check never flips: we=0 → identity, we=1 → delta.rot, zero delta
         * → identity. */
        float nid[4]; mrw_quat_nlerp(MRW_QUAT_IDENTITY, delta[j].rot, we, nid);
        float r[4]; mrw_quat_mul(base[j].rot, nid, r);
        mrw_quat_normalize(r);
        /* trans linear; scale linear toward base·delta (= pose at we=1), no pow. out[j] MAY alias
         * base[j] - base components are read before the matching out write. */
        float ot[3], os[3];
        for (int k = 0; k < 3; ++k) ot[k] = base[j].trans[k] + we * delta[j].trans[k];
        for (int k = 0; k < 3; ++k) {
            float target = base[j].scale[k] * delta[j].scale[k];
            os[k] = base[j].scale[k] + we * (target - base[j].scale[k]);
        }
        for (int k = 0; k < 4; ++k) out[j].rot[k]   = r[k];
        for (int k = 0; k < 3; ++k) out[j].trans[k] = ot[k];
        for (int k = 0; k < 3; ++k) out[j].scale[k] = os[k];
    }
    return MRW_OK;
}
