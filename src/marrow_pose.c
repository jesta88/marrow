/* marrow pose pipeline - sample → local→model → skinning palette, plus root motion.
 * Scalar reference; the SIMD batch path is checked for parity against it. */
#include "marrow_internal.h"
#include "marrow_quat.h"   /* shared scalar quat algebra (also feeds blend/IK) */
#include <math.h>

/* The sampling helpers (mrw_clampf / mrw_mod_pos / mrw_quat_nlerp / mrw_interp_trs /
 * mrw_clip_frame) live in marrow_internal.h so this path and the batch path share one
 * source of truth (the batch is bit-for-bit checked against this code). */

/* ------------------------------------------------------------------ sampling */

mrw_result mrw_clip_sample_local(const mrw_clip_view *clip, float t,
                                       mrw_trs *out_locals, uint32_t joint_capacity) {
    if (!clip || !out_locals) return MRW_E_RANGE;
    if (!mrw_f32_finite(t)) return MRW_E_RANGE;
    uint32_t jc = clip->joint_count;
    if (jc > joint_capacity) return MRW_E_CAPACITY;
    uint32_t i0 = 0; float u = 0.0f;
    int is_static = mrw_clip_frame(clip, t, &i0, &u);
    for (uint32_t j = 0; j < jc; ++j) {
        if (is_static) {
            mrw_clip_sample(clip, j, 0, &out_locals[j]);
        } else {
            mrw_trs a, b;
            mrw_clip_sample(clip, j, i0, &a);
            mrw_clip_sample(clip, j, i0 + 1, &b);
            mrw_interp_trs(&a, &b, u, &out_locals[j]);
        }
    }
    return MRW_OK;
}

/* ------------------------------------------------------------------ local→model→palette */

mrw_result mrw_local_to_model(const mrw_skeleton_view *skel, const mrw_trs *locals,
                                    float *out_model, uint32_t joint_capacity) {
    if (!skel || !locals || !out_model) return MRW_E_RANGE;
    uint32_t n = skel->joint_count;
    if (n > joint_capacity) return MRW_E_CAPACITY;
    for (uint32_t j = 0; j < n; ++j) {
        float local12[12];
        mrw_trs_to_affine(&locals[j], local12);
        if (j == 0) {
            memcpy(out_model, local12, sizeof local12);          /* root: model = local */
        } else {
            uint16_t parent;
            mrw_skeleton_parent(skel, j, &parent);            /* parent < j (validated) */
            mrw_affine_mul(out_model + (size_t)parent * 12, local12, out_model + (size_t)j * 12);
        }
    }
    return MRW_OK;
}

mrw_result mrw_model_to_palette(const mrw_skeleton_view *skel, const float *model,
                                      float *out_palette, uint32_t joint_capacity) {
    if (!skel || !model || !out_palette) return MRW_E_RANGE;
    uint32_t n = skel->joint_count;
    if (n > joint_capacity) return MRW_E_CAPACITY;
    for (uint32_t j = 0; j < n; ++j) {
        float ib[12];
        mrw_skeleton_inverse_bind(skel, j, ib);
        mrw_affine_mul(model + (size_t)j * 12, ib, out_palette + (size_t)j * 12); /* M_j = model ∘ inv_bind */
    }
    return MRW_OK;
}

mrw_result mrw_clip_to_palette(const mrw_skeleton_view *skel, const mrw_clip_view *clip,
                                     float t, float *scratch_model, float *out_palette,
                                     uint32_t joint_capacity) {
    if (!skel || !clip || !scratch_model || !out_palette) return MRW_E_RANGE;
    if (skel->joint_count != clip->joint_count) return MRW_E_INCOMPATIBLE;
    if (!mrw_id_equal(&clip->skeleton_id, &skel->id)) return MRW_E_INCOMPATIBLE;
    if (!mrw_f32_finite(t)) return MRW_E_RANGE;
    if (skel->joint_count > joint_capacity) return MRW_E_CAPACITY;

    uint32_t n = skel->joint_count;
    uint32_t i0 = 0; float u = 0.0f;
    int is_static = mrw_clip_frame(clip, t, &i0, &u);
    for (uint32_t j = 0; j < n; ++j) {
        mrw_trs lj;
        if (is_static) {
            mrw_clip_sample(clip, j, 0, &lj);
        } else {
            mrw_trs a, b;
            mrw_clip_sample(clip, j, i0, &a);
            mrw_clip_sample(clip, j, i0 + 1, &b);
            mrw_interp_trs(&a, &b, u, &lj);
        }
        float local12[12];
        mrw_trs_to_affine(&lj, local12);
        if (j == 0) {
            memcpy(scratch_model, local12, sizeof local12);
        } else {
            uint16_t parent;
            mrw_skeleton_parent(skel, j, &parent);
            mrw_affine_mul(scratch_model + (size_t)parent * 12, local12, scratch_model + (size_t)j * 12);
        }
        float ib[12];
        mrw_skeleton_inverse_bind(skel, j, ib);
        mrw_affine_mul(scratch_model + (size_t)j * 12, ib, out_palette + (size_t)j * 12);
    }
    return MRW_OK;
}

/* ------------------------------------------------------------------ baked decode */

static void baked_decode_bone(const mrw_baked_view *v, uint32_t abs_frame, uint32_t bone, mrw_xform *out) {
    const uint8_t *t0 = v->base + v->texels_off +
        ((size_t)abs_frame * v->frame_stride_texels + (size_t)bone * v->texels_per_bone) * MRW_TEXEL_STRIDE;
    const uint8_t *t1 = t0 + MRW_TEXEL_STRIDE;
    out->rot[0] = mrw_half_to_float(mrw_rd_u16(t0 + 0));
    out->rot[1] = mrw_half_to_float(mrw_rd_u16(t0 + 2));
    out->rot[2] = mrw_half_to_float(mrw_rd_u16(t0 + 4));
    out->rot[3] = mrw_half_to_float(mrw_rd_u16(t0 + 6));
    out->trans[0] = mrw_half_to_float(mrw_rd_u16(t1 + 0));
    out->trans[1] = mrw_half_to_float(mrw_rd_u16(t1 + 2));
    out->trans[2] = mrw_half_to_float(mrw_rd_u16(t1 + 4));
    out->scale    = mrw_half_to_float(mrw_rd_u16(t1 + 6));
    /* renormalize quat after lossy decode */
    float n2 = out->rot[0]*out->rot[0] + out->rot[1]*out->rot[1] + out->rot[2]*out->rot[2] + out->rot[3]*out->rot[3];
    float inv = (n2 > 0.0f) ? 1.0f / sqrtf(n2) : 0.0f;
    for (int k = 0; k < 4; ++k) out->rot[k] *= inv;
}

mrw_result mrw_baked_sample_bone(const mrw_baked_view *baked, uint32_t clip_index,
                                       uint32_t bone, float t, float out_affine12[12]) {
    if (!baked || !out_affine12) return MRW_E_RANGE;
    if (clip_index >= baked->clip_count || bone >= baked->bone_count) return MRW_E_RANGE;
    if (!mrw_f32_finite(t)) return MRW_E_RANGE;

    mrw_baked_clip ce;
    mrw_result rc = mrw_baked_clip_entry(baked, clip_index, &ce);
    if (rc) return rc;

    uint32_t fc = ce.frame_count;
    uint32_t i0 = 0; float u = 0.0f;
    if (fc == 1 || ce.source_duration_s == 0.0f) {
        i0 = 0; u = 0.0f;
    } else {
        int looping = (ce.flags & MRW_BAKED_CLIP_LOOPING) != 0;
        float dur = ce.source_duration_s;
        float t_local = looping ? mrw_mod_pos(t, dur) : mrw_clampf(t, 0.0f, dur);
        float fpos = (t_local / dur) * (float)(fc - 1);
        uint32_t i = (uint32_t)floorf(fpos);
        if (i > fc - 2) i = fc - 2;
        i0 = i; u = fpos - (float)i;
    }

    mrw_xform da, db, d;
    baked_decode_bone(baked, ce.first_frame + i0, bone, &da);
    if (u == 0.0f) {
        d = da;
    } else {
        baked_decode_bone(baked, ce.first_frame + i0 + 1, bone, &db);
        mrw_quat_nlerp(da.rot, db.rot, u, d.rot);
        for (int k = 0; k < 3; ++k) d.trans[k] = da.trans[k] + u * (db.trans[k] - da.trans[k]);
        d.scale = da.scale + u * (db.scale - da.scale);
    }
    mrw_xform_to_affine(&d, out_affine12);
    return MRW_OK;
}

/* ------------------------------------------------------------------ root motion */

typedef struct { float q[4]; float t[3]; } rigid;

/* The quaternion algebra below (Hamilton mul, normalize, conjugate, rotate-vec) is the
 * shared scalar kit from marrow_quat.h. */

/* C = A ∘ B (apply B then A) */
static rigid rigid_compose(const rigid *A, const rigid *B) {
    rigid o;
    mrw_quat_mul(A->q, B->q, o.q);
    mrw_quat_normalize(o.q);
    float rt[3]; mrw_quat_rotate(A->q, B->t, rt);
    for (int k = 0; k < 3; ++k) o.t[k] = rt[k] + A->t[k];
    return o;
}
static rigid rigid_inverse(const rigid *A) {
    rigid o;
    mrw_quat_conj(A->q, o.q);
    mrw_quat_normalize(o.q);
    float rt[3]; mrw_quat_rotate(o.q, A->t, rt);
    for (int k = 0; k < 3; ++k) o.t[k] = -rt[k];
    return o;
}
static rigid rigid_identity(void) { rigid o = { {0,0,0,1}, {0,0,0} }; return o; }
static rigid rigid_pow(rigid base, long long k) {
    if (k < 0) { base = rigid_inverse(&base); k = -k; }
    rigid result = rigid_identity();
    while (k > 0) {
        if (k & 1) result = rigid_compose(&result, &base);
        base = rigid_compose(&base, &base);
        k >>= 1;
    }
    return result;
}
static rigid xform_to_rigid(const mrw_xform *x) {
    rigid r;
    for (int k = 0; k < 4; ++k) r.q[k] = x->rot[k];
    for (int k = 0; k < 3; ++k) r.t[k] = x->trans[k];
    return r;
}

/* Sample the rigid trajectory R(tau) for tau in [0,T] (non-looping clamp). */
static rigid sample_root(const mrw_clip_view *clip, float tau, float T) {
    uint32_t n = clip->sample_count;
    float tl = mrw_clampf(tau, 0.0f, T);
    float fpos = (T > 0.0f) ? (tl / T) * (float)(n - 1) : 0.0f;
    uint32_t i = (uint32_t)floorf(fpos);
    if (i > n - 2) i = n - 2;
    float u = fpos - (float)i;
    mrw_xform a, b;
    mrw_clip_root_sample(clip, i, &a);
    mrw_clip_root_sample(clip, i + 1, &b);
    rigid o;
    mrw_quat_nlerp(a.rot, b.rot, u, o.q);
    for (int k = 0; k < 3; ++k) o.t[k] = a.trans[k] + u * (b.trans[k] - a.trans[k]);
    return o;
}

/* Accumulated trajectory U(t). */
static rigid root_U(const mrw_clip_view *clip, const rigid *invR0, const rigid *C, float T, int looping, float t) {
    if (looping) {
        double kd = floor((double)t / (double)T);
        double taud = (double)t - kd * (double)T;        /* ideally in [0,T) */
        /* Normalize the (k,tau) PAIR together - float rounding can push tau to exactly T
         * or just below 0; resetting tau alone (without moving k) would jump U by a whole
         * cycle near the loop boundary. Move the cycle with tau so U stays continuous. */
        if (taud >= (double)T) { taud -= (double)T; kd += 1.0; }
        else if (taud < 0.0)   { taud += (double)T; kd -= 1.0; }
        if (taud < 0.0) taud = 0.0; else if (taud >= (double)T) taud = 0.0;
        /* Saturate the integer cycle count so the cast can never be UB. This bounds |k| to
         * ~9e18; for |t| that large a binary32 has lost all sub-cycle precision anyway, so
         * the per-cycle index is already meaningless - a documented, deliberate limit. */
        long long k = (kd >  9.0e18) ?  9000000000000000000LL
                    : (kd < -9.0e18) ? -9000000000000000000LL
                    : (long long)kd;
        float tau = (float)taud;
        rigid Rt = sample_root(clip, tau, T);
        rigid Ptau = rigid_compose(invR0, &Rt);
        rigid Ck = rigid_pow(*C, k);
        return rigid_compose(&Ck, &Ptau);
    } else {
        float tc = mrw_clampf(t, 0.0f, T);
        rigid Rt = sample_root(clip, tc, T);
        return rigid_compose(invR0, &Rt);
    }
}

mrw_result mrw_root_motion(const mrw_clip_view *clip, float t0, float t1, mrw_xform *out_delta) {
    if (!clip || !out_delta) return MRW_E_RANGE;
    if (!mrw_f32_finite(t0) || !mrw_f32_finite(t1)) return MRW_E_RANGE;

    /* identity delta */
    out_delta->rot[0] = 0; out_delta->rot[1] = 0; out_delta->rot[2] = 0; out_delta->rot[3] = 1;
    out_delta->trans[0] = 0; out_delta->trans[1] = 0; out_delta->trans[2] = 0;
    out_delta->scale = 1.0f;

    if (!(clip->flags & MRW_CLIP_HAS_ROOT_MOTION)) return MRW_OK;
    uint32_t n = clip->sample_count;
    if (n == 1) return MRW_OK;
    if (t0 == t1) return MRW_OK;

    float T = (float)(n - 1) / clip->fps;
    if (!(T > 0.0f)) return MRW_OK;   /* degenerate duration (e.g. fps underflow) ⇒ identity */
    int looping = (clip->flags & MRW_CLIP_LOOPING) != 0;

    mrw_xform r0x, rlastx;
    mrw_clip_root_sample(clip, 0, &r0x);
    mrw_clip_root_sample(clip, n - 1, &rlastx);
    rigid R0 = xform_to_rigid(&r0x), Rlast = xform_to_rigid(&rlastx);
    rigid invR0 = rigid_inverse(&R0);
    rigid C = rigid_compose(&invR0, &Rlast);   /* per-cycle delta C = P(T) */

    rigid U0 = root_U(clip, &invR0, &C, T, looping, t0);
    rigid U1 = root_U(clip, &invR0, &C, T, looping, t1);
    rigid invU0 = rigid_inverse(&U0);
    rigid D = rigid_compose(&invU0, &U1);      /* D = U(t0)⁻¹ · U(t1) */

    for (int k = 0; k < 4; ++k) out_delta->rot[k] = D.q[k];
    for (int k = 0; k < 3; ++k) out_delta->trans[k] = D.t[k];
    out_delta->scale = 1.0f;
    return MRW_OK;
}
