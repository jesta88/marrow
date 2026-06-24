/* See convert.h. The conversion is a fold + resample: glTF composes the full node tree (including
 * non-joint nodes between/above joints), marrow composes only joint→joint locals, so each marrow
 * local is the product of node-locals along the path from the parent joint (exclusive) to the
 * joint (inclusive) - evaluated per sample for clips, at the default pose for rest_local. glTF and
 * marrow share conventions (right-handed, +Y up, meters/seconds/radians, quat (x,y,z,w), local
 * compose T·R·S), so there is no axis/unit conversion. */
#include "convert.h"
#include "joint_order.h"     /* mrw_g2m_joint_order: shared skin-order → marrow-index DFS */
#include "mrw_authoring.h"
#include "mrw_decompose.h"   /* mrw_mat3_to_quat (matrix→quat for the fold decompose) */
#include "cgltf.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define G2M_MAX_FOLD_DEPTH 256

static void set_diag(char *diag, size_t cap, const char *fmt, ...) {
    if (!diag || cap == 0) return;
    va_list ap; va_start(ap, fmt);
    vsnprintf(diag, cap, fmt, ap);
    va_end(ap);
}

/* ---- small affine helpers (3×4 row-major [A|t], column vectors p'=M·[p,1]) ---- */
static const float G2M_IDENT12[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};

/* glTF column-major 4×4 → marrow 3×4 row-major: row r = (m[r], m[4+r], m[8+r], m[12+r]); the
 * dropped 4th row is the implicit (0,0,0,1). Used for node `matrix` and inverse-bind matrices. */
static void colmajor4x4_to_affine(const float m[16], float out12[12]) {
    for (int r = 0; r < 3; ++r) {
        out12[r*4+0] = m[0+r];  out12[r*4+1] = m[4+r];
        out12[r*4+2] = m[8+r];  out12[r*4+3] = m[12+r];
    }
}

static void quat_normalize(float q[4]) {
    float n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    if (n2 > 0.0f) { float inv = 1.0f / sqrtf(n2); for (int k = 0; k < 4; ++k) q[k] *= inv; }
    else { q[0] = q[1] = q[2] = 0.0f; q[3] = 1.0f; }
}

/* Decompose a 3×4 affine into a TRS with NON-uniform scale (local = T·R·S, column-scale
 * convention): s_k = ‖column k‖, R column = column/s_k, then quaternion via Shepperd. Reflections
 * (det R < 0) are carried as a negative s[0] so R stays proper-orthogonal. Returns 0 (shear/
 * degenerate, not representable as TRS) when a scale column is near-zero or the normalized columns
 * are not orthogonal within `tol`. */
static int decompose_trs(const float m[12], float q[4], float t[3], float s[3], float tol) {
    float c[3][3]; /* c[k] = column k = (m[k], m[4+k], m[8+k]) */
    for (int k = 0; k < 3; ++k) { c[k][0] = m[k]; c[k][1] = m[4+k]; c[k][2] = m[8+k]; }
    t[0] = m[3]; t[1] = m[7]; t[2] = m[11];
    for (int i = 0; i < 12; ++i) if (!isfinite(m[i])) return 0;

    float R[9];
    for (int k = 0; k < 3; ++k) {
        float len = sqrtf(c[k][0]*c[k][0] + c[k][1]*c[k][1] + c[k][2]*c[k][2]);
        if (!(len > 1e-8f)) return 0; /* degenerate / near-zero scale column */
        s[k] = len;
        float inv = 1.0f / len;
        R[0*3+k] = c[k][0]*inv; R[1*3+k] = c[k][1]*inv; R[2*3+k] = c[k][2]*inv;
    }
    /* shear test: normalized columns must be mutually orthogonal */
    for (int i = 0; i < 3; ++i) for (int j = i+1; j < 3; ++j) {
        float d = R[0*3+i]*R[0*3+j] + R[1*3+i]*R[1*3+j] + R[2*3+i]*R[2*3+j];
        if (fabsf(d) > tol) return 0;
    }
    float det = R[0]*(R[4]*R[8]-R[5]*R[7]) - R[1]*(R[3]*R[8]-R[5]*R[6]) + R[2]*(R[3]*R[7]-R[4]*R[6]);
    if (det < 0.0f) { /* reflection → fold into a negative scale on column 0 */
        s[0] = -s[0];
        R[0] = -R[0]; R[3] = -R[3]; R[6] = -R[6];
    }
    mrw_mat3_to_quat(R, q);
    return 1;
}

/* ---- sampler evaluation (per glTF sampler.interpolation; clamps outside [t0,tN]) ---- */
static void lerp_n(const float a[4], const float b[4], float u, float out[4], int n) {
    for (int k = 0; k < n; ++k) out[k] = a[k] + u * (b[k] - a[k]);
}
/* hemisphere-corrected slerp; near-parallel pairs fall back to normalized lerp (avoid 1/sinθ). */
static void quat_slerp(const float a[4], const float b0[4], float u, float out[4]) {
    float b[4]; float dot = a[0]*b0[0] + a[1]*b0[1] + a[2]*b0[2] + a[3]*b0[3];
    for (int k = 0; k < 4; ++k) b[k] = b0[k];
    if (dot < 0.0f) { for (int k = 0; k < 4; ++k) b[k] = -b[k]; dot = -dot; }
    if (dot > 0.999999f) { lerp_n(a, b, u, out, 4); quat_normalize(out); return; }
    float theta = acosf(dot), st = sinf(theta);
    float wa = sinf((1.0f - u) * theta) / st, wb = sinf(u * theta) / st;
    for (int k = 0; k < 4; ++k) out[k] = wa*a[k] + wb*b[k];
    quat_normalize(out);
}

static int read_elem(const cgltf_accessor *acc, cgltf_size idx, float *out, int n) {
    return cgltf_accessor_read_float(acc, idx, out, (cgltf_size)n) ? 0 : -1;
}

/* Evaluate one sampler at time t into out[n] (n=3 T/S, 4 R). is_rotation drives slerp + normalize.
 * Returns 0 on success, -1 on an accessor read error. */
static int eval_sampler(const cgltf_animation_sampler *smp, int n, int is_rotation, float t, float out[4]) {
    cgltf_size kc = smp->input->count;
    int cubic = smp->interpolation == cgltf_interpolation_type_cubic_spline;
    if (kc == 0) return -1;

    float ta, tb;
    if (read_elem(smp->input, 0, &ta, 1) || read_elem(smp->input, kc - 1, &tb, 1)) return -1;

    /* clamp to ends (glTF sampler clamping); CUBICSPLINE end value is the middle (value) triple */
    if (kc == 1 || t <= ta) {
        if (read_elem(smp->output, cubic ? 1 : 0, out, n)) return -1;
        if (is_rotation) quat_normalize(out);
        return 0;
    }
    if (t >= tb) {
        cgltf_size last = kc - 1;
        if (read_elem(smp->output, cubic ? last*3 + 1 : last, out, n)) return -1;
        if (is_rotation) quat_normalize(out);
        return 0;
    }

    /* find segment [tk, tk1] with tk <= t < tk1 */
    cgltf_size k = 0; float tk = ta, tk1 = tb;
    for (cgltf_size i = 0; i + 1 < kc; ++i) {
        float a, b;
        if (read_elem(smp->input, i, &a, 1) || read_elem(smp->input, i + 1, &b, 1)) return -1;
        if (t >= a && t <= b) { k = i; tk = a; tk1 = b; break; }
    }
    float dt = tk1 - tk;
    float u = (dt > 0.0f) ? (t - tk) / dt : 0.0f;

    if (smp->interpolation == cgltf_interpolation_type_step) {
        if (read_elem(smp->output, cubic ? k*3 + 1 : k, out, n)) return -1; /* step never cubic, but be safe */
        if (is_rotation) quat_normalize(out);
        return 0;
    }
    if (!cubic) { /* LINEAR */
        float v0[4], v1[4];
        if (read_elem(smp->output, k, v0, n) || read_elem(smp->output, k + 1, v1, n)) return -1;
        if (is_rotation) quat_slerp(v0, v1, u, out);
        else             lerp_n(v0, v1, u, out, n);
        return 0;
    }
    /* CUBICSPLINE Hermite: each key is (inTangent, value, outTangent); tangents scaled by dt. */
    float v0[4], b0[4], a1[4], v1[4];
    if (read_elem(smp->output, k*3 + 1, v0, n) || read_elem(smp->output, k*3 + 2, b0, n) ||
        read_elem(smp->output, (k+1)*3 + 0, a1, n) || read_elem(smp->output, (k+1)*3 + 1, v1, n)) return -1;
    float u2 = u*u, u3 = u2*u;
    float h00 = 2*u3 - 3*u2 + 1, h10 = u3 - 2*u2 + u, h01 = -2*u3 + 3*u2, h11 = u3 - u2;
    for (int kk = 0; kk < n; ++kk)
        out[kk] = h00*v0[kk] + h10*dt*b0[kk] + h01*v1[kk] + h11*dt*a1[kk];
    if (is_rotation) quat_normalize(out);
    return 0;
}

/* ---- per-animation sampler lookup (NULL = rest pose / unanimated) ---- */
typedef struct {
    const cgltf_animation_sampler **t, **r, **s; /* each sized data->nodes_count, NULL entries unanimated */
} samplers;

/* node's local TRS at time `t` (animated paths from ss, others from node defaults). */
static int node_trs_at(cgltf_data *data, const cgltf_node *node, const samplers *ss, float t, mrw_trs *out) {
    uint32_t ni = (uint32_t)(node - data->nodes);
    const cgltf_animation_sampler *st = ss ? ss->t[ni] : NULL;
    const cgltf_animation_sampler *sr = ss ? ss->r[ni] : NULL;
    const cgltf_animation_sampler *ss_ = ss ? ss->s[ni] : NULL;

    if (sr) { if (eval_sampler(sr, 4, 1, t, out->rot)) return -1; }
    else { if (node->has_rotation) memcpy(out->rot, node->rotation, sizeof out->rot);
           else { out->rot[0]=out->rot[1]=out->rot[2]=0.0f; out->rot[3]=1.0f; } }
    quat_normalize(out->rot);

    if (st) { float v[4]; if (eval_sampler(st, 3, 0, t, v)) return -1; out->trans[0]=v[0]; out->trans[1]=v[1]; out->trans[2]=v[2]; }
    else if (node->has_translation) memcpy(out->trans, node->translation, sizeof out->trans);
    else { out->trans[0]=out->trans[1]=out->trans[2]=0.0f; }

    if (ss_) { float v[4]; if (eval_sampler(ss_, 3, 0, t, v)) return -1; out->scale[0]=v[0]; out->scale[1]=v[1]; out->scale[2]=v[2]; }
    else if (node->has_scale) memcpy(out->scale, node->scale, sizeof out->scale);
    else { out->scale[0]=out->scale[1]=out->scale[2]=1.0f; }
    return 0;
}

/* node's local transform at time `t` as a 3×4 affine (static `matrix` nodes are never animated). */
static int node_local_affine(cgltf_data *data, const cgltf_node *node, const samplers *ss, float t, float out12[12]) {
    if (node->has_matrix) { colmajor4x4_to_affine(node->matrix, out12); return 0; }
    mrw_trs trs;
    if (node_trs_at(data, node, ss, t, &trs)) return -1;
    mrw_trs_to_affine(&trs, out12);
    return 0;
}

/* ---- skeleton context shared by rest + clip folds ---- */
typedef struct {
    cgltf_data *data;
    const cgltf_skin *skin;
    const int32_t  *parent_js; /* skin-order joint -> parent skin-order index, -1 = root */
    const uint32_t *order;     /* marrow index -> skin-order joint index                 */
    uint32_t jc;
    char *diag; size_t diag_cap;
} skelctx;

/* Fold the node chain from the parent joint (exclusive) to joint `mj` (inclusive) into a marrow
 * local TRS at time `t` (ss=NULL → rest pose). Fast path: a single non-`matrix` node emits its TRS
 * directly; otherwise the path affines are composed and decomposed (shear ⇒ error). */
static mrw_result fold_local(const skelctx *c, const samplers *ss, float t, uint32_t mj, float out10[10]) {
    uint32_t js = c->order[mj];
    const cgltf_node *N = c->skin->joints[js];
    const cgltf_node *PN = (c->parent_js[js] < 0) ? NULL : c->skin->joints[c->parent_js[js]];

    const cgltf_node *path[G2M_MAX_FOLD_DEPTH];
    int n = 0;
    for (const cgltf_node *cur = N; cur != PN && cur != NULL; cur = cur->parent) {
        if (n >= G2M_MAX_FOLD_DEPTH) { set_diag(c->diag, c->diag_cap, "joint %u: fold path deeper than %d", mj, G2M_MAX_FOLD_DEPTH); return MRW_E_FORMAT; }
        path[n++] = cur;
    }

    if (n == 1 && !N->has_matrix) { /* fast path: exact TRS, no affine round-trip */
        mrw_trs trs;
        if (node_trs_at(c->data, N, ss, t, &trs)) { set_diag(c->diag, c->diag_cap, "joint %u: sampler read failed", mj); return MRW_E_FORMAT; }
        memcpy(out10 + 0, trs.rot, 16);
        memcpy(out10 + 4, trs.trans, 12);
        memcpy(out10 + 7, trs.scale, 12);
        return MRW_OK;
    }

    /* general: compose parent-most → joint, then decompose to TRS */
    float acc[12]; memcpy(acc, G2M_IDENT12, sizeof acc);
    for (int i = n - 1; i >= 0; --i) {
        float L[12], tmp[12];
        if (node_local_affine(c->data, path[i], ss, t, L)) { set_diag(c->diag, c->diag_cap, "joint %u: sampler read failed", mj); return MRW_E_FORMAT; }
        mrw_affine_mul(acc, L, tmp);
        memcpy(acc, tmp, sizeof acc);
    }
    float q[4], tr[3], sc[3];
    if (!decompose_trs(acc, q, tr, sc, 1e-4f)) {
        set_diag(c->diag, c->diag_cap, "joint %u: folded transform has shear or degenerate scale (not representable as TRS)", mj);
        return MRW_E_FORMAT;
    }
    memcpy(out10 + 0, q, 16);
    memcpy(out10 + 4, tr, 12);
    memcpy(out10 + 7, sc, 12);
    return MRW_OK;
}

/* mark every node that appears in any joint's fold path (the nodes whose animation affects output) */
static void mark_relevant(const skelctx *c, uint8_t *relevant) {
    for (uint32_t mj = 0; mj < c->jc; ++mj) {
        uint32_t js = c->order[mj];
        const cgltf_node *N = c->skin->joints[js];
        const cgltf_node *PN = (c->parent_js[js] < 0) ? NULL : c->skin->joints[c->parent_js[js]];
        for (const cgltf_node *cur = N; cur != PN && cur != NULL; cur = cur->parent)
            relevant[(uint32_t)(cur - c->data->nodes)] = 1;
    }
}

/* ---- one clip: resample an animation onto a dense joint-major codec-0 track ---- */
static mrw_result build_clip(const skelctx *c, const cgltf_animation *anim, float fps_nominal, int loop,
                             int force_codec0, const uint8_t *relevant, mrw_clip *out_clip, float **out_samples) {
    cgltf_data *data = c->data;
    cgltf_size nn = data->nodes_count;
    const char *aname = anim->name ? anim->name : "(unnamed)";

    samplers ss;
    ss.t = (const cgltf_animation_sampler **)calloc(nn, sizeof(void *));
    ss.r = (const cgltf_animation_sampler **)calloc(nn, sizeof(void *));
    ss.s = (const cgltf_animation_sampler **)calloc(nn, sizeof(void *));
    if (!ss.t || !ss.r || !ss.s) { free((void*)ss.t); free((void*)ss.r); free((void*)ss.s); set_diag(c->diag, c->diag_cap, "out of memory"); return MRW_E_OVERFLOW; }

    mrw_result rc = MRW_OK;
    float D = 0.0f;
    int warned_weights = 0;
    for (cgltf_size ci = 0; ci < anim->channels_count; ++ci) {
        const cgltf_animation_channel *ch = &anim->channels[ci];
        if (!ch->target_node) continue;
        uint32_t ni = (uint32_t)(ch->target_node - data->nodes);
        if (ni >= nn || !relevant[ni]) continue; /* doesn't drive this skeleton */
        if (ch->target_path == cgltf_animation_path_type_weights) {
            if (!warned_weights) { fprintf(stderr, "gltf2marrow: note: animation '%s' has morph-weight channels; ignored.\n", aname); warned_weights = 1; }
            continue;
        }
        const cgltf_animation_sampler ***slot =
            (ch->target_path == cgltf_animation_path_type_translation) ? &ss.t :
            (ch->target_path == cgltf_animation_path_type_rotation)    ? &ss.r :
            (ch->target_path == cgltf_animation_path_type_scale)       ? &ss.s : NULL;
        if (!slot) continue;
        if ((*slot)[ni]) { set_diag(c->diag, c->diag_cap, "animation '%s': duplicate channel for one node+path", aname); rc = MRW_E_FORMAT; goto cleanup; }
        (*slot)[ni] = ch->sampler;
        if (ch->sampler && ch->sampler->input && ch->sampler->input->count > 0) {
            float last; if (!read_elem(ch->sampler->input, ch->sampler->input->count - 1, &last, 1) && isfinite(last) && last > D) D = last;
        }
    }

    /* sample_count derivation, range-checked before the float→uint cast (a huge or non-finite
     * D·fps would make the cast UB and the allocation absurd; glTF key times are untrusted). */
    if (!isfinite(D) || D < 0.0f) { set_diag(c->diag, c->diag_cap, "animation '%s': non-finite duration", aname); rc = MRW_E_RANGE; goto cleanup; }
    uint32_t sample_count;
    if (D <= 0.0f) {
        sample_count = 1u;
    } else {
        double sc = floor((double)D * (double)fps_nominal + 0.5) + 1.0;
        if (!(sc >= 1.0 && sc <= (double)0xFFFFFFFFu)) { set_diag(c->diag, c->diag_cap, "animation '%s': duration*fps out of range (%g samples)", aname, sc); rc = MRW_E_RANGE; goto cleanup; }
        sample_count = (uint32_t)sc;
    }
    float fps_stored = (sample_count >= 2) ? (float)(sample_count - 1) / D : fps_nominal;

    uint32_t flags = 0;
    if (loop) {
        if (sample_count >= 2) flags |= MRW_CLIP_LOOPING;
        else fprintf(stderr, "gltf2marrow: note: animation '%s' is static; --loop ignored for it.\n", aname);
    }
    fprintf(stderr, "gltf2marrow: clip '%s' - duration=%.4gs, samples=%u, fps nominal=%.4g stored=%.6g\n",
            aname, (double)D, sample_count, (double)fps_nominal, (double)fps_stored);

    float *samples = (float *)malloc((size_t)c->jc * sample_count * 10 * sizeof(float));
    if (!samples) { set_diag(c->diag, c->diag_cap, "out of memory (clip samples)"); rc = MRW_E_OVERFLOW; goto cleanup; }

    for (uint32_t sidx = 0; sidx < sample_count; ++sidx) {
        /* last sample lands on exactly D (the float division (n-1)/fps_stored may drift); the
         * resampler clamps the final sample to the source duration. */
        float t = (sample_count < 2) ? 0.0f : (sidx == sample_count - 1 ? D : (float)sidx / fps_stored);
        for (uint32_t mj = 0; mj < c->jc; ++mj) {
            float *dst = samples + ((size_t)mj * sample_count + sidx) * 10; /* joint-major layout */
            rc = fold_local(c, &ss, t, mj, dst);
            if (rc) { free(samples); goto cleanup; }
        }
    }

    if (flags & MRW_CLIP_LOOPING) { /* warn if the track does not close (loop wrap exposes the seam) */
        for (uint32_t mj = 0; mj < c->jc; ++mj) {
            const float *a = samples + ((size_t)mj * sample_count + 0) * 10;
            const float *b = samples + ((size_t)mj * sample_count + (sample_count - 1)) * 10;
            float d = 0.0f; for (int k = 0; k < 10; ++k) { float e = fabsf(a[k] - b[k]); if (e > d) d = e; }
            if (d > 1e-4f) { fprintf(stderr, "gltf2marrow: note: clip '%s' --loop set but joint %u does not close (Δ=%.3g).\n", aname, mj, (double)d); break; }
        }
    }

    /* codec-1 eligibility (tol 1e-4): if EVERY sample's scale is within 1e-4 of 1 across every joint,
     * emit the scale-free codec. This SNAPS each scale to exactly 1.0f - a bounded (≤1e-4), lossy snap
     * of the scale channel only (rotation + translation bytes are preserved) - so the on-disk q4+t3 and
     * the clip id over the snapped 10-float stream are canonical. Otherwise codec 0 (raw TRS). The
     * runtime only consumes either codec; it never re-encodes. */
    uint32_t codec = force_codec0 ? 0 : 1;
    size_t nsamp = (size_t)c->jc * sample_count;
    for (size_t r = 0; r < nsamp && codec; ++r) {
        const float *s = samples + r * 10 + 7;
        if (fabsf(s[0] - 1.0f) > 1e-4f || fabsf(s[1] - 1.0f) > 1e-4f || fabsf(s[2] - 1.0f) > 1e-4f) codec = 0;
    }
    if (codec) {
        for (size_t r = 0; r < nsamp; ++r) { float *s = samples + r * 10 + 7; s[0] = s[1] = s[2] = 1.0f; }
        fprintf(stderr, "gltf2marrow: clip '%s' - unit scale ⇒ codec 1 (scale-free, 28 B/sample, −30%% clip).\n", aname);
    } else if (force_codec0) {
        fprintf(stderr, "gltf2marrow: clip '%s' - --codec0 forces raw codec 0 (40 B/sample), skipping the unit-scale snap.\n", aname);
    }

    out_clip->fps = fps_stored;
    out_clip->sample_count = sample_count;
    out_clip->flags = flags;
    out_clip->samples = samples;
    out_clip->root_track = NULL;
    out_clip->codec = codec;
    *out_samples = samples;

cleanup:
    free((void*)ss.t); free((void*)ss.r); free((void*)ss.s);
    return rc;
}

/* ---- self-validation: re-open the produced blob through the runtime validator ---- */
static mrw_result self_validate(const uint8_t *buf, size_t size, char *diag, size_t diag_cap) {
    mrw_blob blob;
    mrw_result r = mrw_blob_open(buf, (uint64_t)size, &blob);
    if (r) { set_diag(diag, diag_cap, "self-validation: mrw_blob_open rejected the output (%d)", (int)r); return r; }
    for (uint32_t i = 0; i < blob.section_count; ++i) {
        uint32_t ty = 0;
        r = mrw_blob_section_type(&blob, i, &ty);
        if (r) { set_diag(diag, diag_cap, "self-validation: section %u type unreadable (%d)", i, (int)r); return r; }
        if (ty == MRW_SECTION_SKELETON) {
            mrw_skeleton_view sv; r = mrw_skeleton_view_at(&blob, i, &sv);
            if (r) { set_diag(diag, diag_cap, "self-validation: skeleton section %u invalid (%d)", i, (int)r); return r; }
        } else if (ty == MRW_SECTION_CLIP) {
            mrw_clip_view cv; r = mrw_clip_view_at(&blob, i, &cv);
            if (r) { set_diag(diag, diag_cap, "self-validation: clip section %u invalid (%d)", i, (int)r); return r; }
        }
    }
    return MRW_OK;
}

mrw_result mrw_g2m_convert(const mrw_g2m_options *opt, uint8_t **out_buf, size_t *out_size,
                           char *diag, size_t diag_cap) {
    if (!opt || !opt->input_path || !out_buf || !out_size) { set_diag(diag, diag_cap, "invalid arguments"); return MRW_E_FORMAT; }
    float fps_nominal = (opt->fps > 0.0f) ? opt->fps : 30.0f;

    cgltf_options copt = {0};
    cgltf_data *data = NULL;
    cgltf_result cr = cgltf_parse_file(&copt, opt->input_path, &data);
    if (cr != cgltf_result_success) { set_diag(diag, diag_cap, "cgltf parse failed (%d): %s", (int)cr, opt->input_path); return MRW_E_FORMAT; }
    cr = cgltf_load_buffers(&copt, data, opt->input_path);
    if (cr != cgltf_result_success) { set_diag(diag, diag_cap, "cgltf buffer load failed (%d)", (int)cr); cgltf_free(data); return MRW_E_FORMAT; }
    /* structural validation (accessor bounds, sampler input/output presence/types, buffer-view
     * ranges) - so the resampler can trust every accessor/sampler pointer it later dereferences. */
    cr = cgltf_validate(data);
    if (cr != cgltf_result_success) { set_diag(diag, diag_cap, "glTF failed validation (%d): malformed accessor/sampler", (int)cr); cgltf_free(data); return MRW_E_FORMAT; }

    mrw_result rc = MRW_OK;
    mrw_g2m_joint_order jo = {0};
    uint16_t *parent = NULL; float *rest_local = NULL, *inverse_bind = NULL;
    char **names = NULL; char *name_pool = NULL;
    uint8_t *relevant = NULL;
    mrw_clip *clips = NULL; float **clip_samples = NULL; uint32_t nclip = 0;
    uint32_t *anim_sel = NULL;
    uint8_t *buf = NULL; size_t bufsz = 0;

    /* ---- select skin ---- */
    if (data->skins_count == 0) { set_diag(diag, diag_cap, "no skin in glTF (nothing to import)"); rc = MRW_E_FORMAT; goto done; }
    cgltf_size skin_i;
    if (opt->skin_index >= 0) {
        if ((cgltf_size)opt->skin_index >= data->skins_count) { set_diag(diag, diag_cap, "--skin %d out of range (%zu skins)", opt->skin_index, data->skins_count); rc = MRW_E_RANGE; goto done; }
        skin_i = (cgltf_size)opt->skin_index;
    } else if (data->skins_count == 1) {
        skin_i = 0;
    } else { set_diag(diag, diag_cap, "%zu skins present; pass --skin <index>", data->skins_count); rc = MRW_E_FORMAT; goto done; }
    const cgltf_skin *skin = &data->skins[skin_i];

    /* ---- parent resolution + single-root requirement + topological (marrow) order ---- */
    rc = mrw_g2m_joint_order_build(skin, &jo, diag, diag_cap);
    if (rc) goto done;
    uint32_t jc = jo.joint_count;

    skelctx ctx = { data, skin, jo.parent_js, jo.order, jc, diag, diag_cap };

    /* ---- skeleton arrays in marrow order ---- */
    parent = (uint16_t *)malloc((size_t)jc * sizeof(uint16_t));
    rest_local = (float *)malloc((size_t)jc * 10 * sizeof(float));
    inverse_bind = (float *)malloc((size_t)jc * 12 * sizeof(float));
    names = (char **)malloc((size_t)jc * sizeof(char *));
    name_pool = (char *)malloc((size_t)jc * 24); /* "joint_4294967295" + NUL fits in 24 */
    if (!parent || !rest_local || !inverse_bind || !names || !name_pool) { rc = MRW_E_OVERFLOW; goto done; }

    for (uint32_t mj = 0; mj < jc; ++mj) {
        uint32_t js = jo.order[mj];
        parent[mj] = (jo.parent_js[js] < 0) ? 0xFFFFu : (uint16_t)jo.marrow_of[jo.parent_js[js]];

        rc = fold_local(&ctx, NULL, 0.0f, mj, rest_local + (size_t)mj * 10);
        if (rc) goto done;

        if (skin->inverse_bind_matrices) {
            float m16[16];
            if (!cgltf_accessor_read_float(skin->inverse_bind_matrices, js, m16, 16)) { set_diag(diag, diag_cap, "joint %u: inverse-bind matrix read failed", mj); rc = MRW_E_FORMAT; goto done; }
            colmajor4x4_to_affine(m16, inverse_bind + (size_t)mj * 12);
        } else {
            memcpy(inverse_bind + (size_t)mj * 12, G2M_IDENT12, sizeof G2M_IDENT12);
        }

        const char *nm = skin->joints[js]->name;
        if (nm) { names[mj] = (char *)nm; }
        else { char *p = name_pool + (size_t)mj * 24; snprintf(p, 24, "joint_%u", mj); names[mj] = p; }
    }

    /* ---- select animations ---- */
    relevant = (uint8_t *)calloc(data->nodes_count, 1);
    if (!relevant) { rc = MRW_E_OVERFLOW; goto done; }
    mark_relevant(&ctx, relevant);

    /* worst case is one selection per requested name (named) or per animation (all); a repeated
     * --anim must not push past the allocation, so size by the request and dedup indices. */
    size_t sel_cap = (opt->anims && opt->anim_count > 0) ? opt->anim_count : data->animations_count;
    anim_sel = (uint32_t *)malloc((sel_cap ? sel_cap : 1) * sizeof(uint32_t));
    if (!anim_sel) { rc = MRW_E_OVERFLOW; goto done; }
    if (opt->anims && opt->anim_count > 0) {
        for (uint32_t a = 0; a < opt->anim_count; ++a) {
            uint32_t found = 0;
            for (cgltf_size i = 0; i < data->animations_count; ++i)
                if (data->animations[i].name && strcmp(data->animations[i].name, opt->anims[a]) == 0) {
                    uint32_t dup = 0; for (uint32_t k = 0; k < nclip; ++k) if (anim_sel[k] == (uint32_t)i) { dup = 1; break; }
                    if (!dup) anim_sel[nclip++] = (uint32_t)i;
                    found = 1; break;
                }
            if (!found) { set_diag(diag, diag_cap, "--anim '%s' not found", opt->anims[a]); rc = MRW_E_FORMAT; goto done; }
        }
    } else {
        for (cgltf_size i = 0; i < data->animations_count; ++i) anim_sel[nclip++] = (uint32_t)i;
    }

    /* ---- build clips ---- */
    if (nclip) {
        clips = (mrw_clip *)calloc(nclip, sizeof(mrw_clip));
        clip_samples = (float **)calloc(nclip, sizeof(float *));
        if (!clips || !clip_samples) { rc = MRW_E_OVERFLOW; goto done; }
        for (uint32_t ci = 0; ci < nclip; ++ci) {
            rc = build_clip(&ctx, &data->animations[anim_sel[ci]], fps_nominal, opt->loop, opt->force_codec0, relevant, &clips[ci], &clip_samples[ci]);
            if (rc) goto done;
        }
    }

    /* ---- serialize ---- */
    mrw_skel ms = { jc, parent, rest_local, inverse_bind, (const char **)names };
    rc = mrw_authoring_build(&ms, clips, nclip, NULL, &buf, &bufsz);
    if (rc) { set_diag(diag, diag_cap, "authoring build failed (%d)", (int)rc); goto done; }

    /* ---- self-validate before returning (never emit a malformed .mrw) ---- */
    rc = self_validate(buf, bufsz, diag, diag_cap);
    if (rc) { mrw_authoring_free(buf); buf = NULL; goto done; }

    *out_buf = buf; *out_size = bufsz; buf = NULL; /* ownership transferred */
    rc = MRW_OK;

done:
    if (buf) mrw_authoring_free(buf);
    if (clip_samples) { for (uint32_t i = 0; i < nclip; ++i) free(clip_samples[i]); free(clip_samples); }
    free(clips);
    free(anim_sel); free(relevant);
    free(name_pool); free(names); free(inverse_bind); free(rest_local); free(parent);
    mrw_g2m_joint_order_free(&jo);
    cgltf_free(data);
    return rc;
}
