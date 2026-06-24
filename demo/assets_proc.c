#include "assets_proc.h"

#include "marrow.h"
#include "mrw_authoring.h"
#include "mrw_bake.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ biped skeleton (14 joints) */

enum {
    J_PELVIS, J_SPINE, J_CHEST, J_HEAD,
    J_LUARM, J_LLARM, J_RUARM, J_RLARM,
    J_LTHIGH, J_LSHIN, J_LFOOT, J_RTHIGH, J_RSHIN, J_RFOOT,
    J_COUNT
};

static const uint16_t k_parent[J_COUNT] = {
    0xFFFF, J_PELVIS, J_SPINE, J_CHEST,
    J_CHEST, J_LUARM, J_CHEST, J_RUARM,
    J_PELVIS, J_LTHIGH, J_LSHIN, J_PELVIS, J_RTHIGH, J_RSHIN
};
static const char *k_names[J_COUNT] = {
    "pelvis", "spine", "chest", "head",
    "L_upperarm", "L_forearm", "R_upperarm", "R_forearm",
    "L_thigh", "L_shin", "L_foot", "R_thigh", "R_shin", "R_foot"
};
/* local rest translation (m); all rest rotations identity, scale 1 */
static const float k_rest_t[J_COUNT][3] = {
    {  0.00f, 0.95f,  0.00f },  /* pelvis (root, in model space) */
    {  0.00f, 0.18f,  0.00f },  /* spine  */
    {  0.00f, 0.22f,  0.00f },  /* chest  */
    {  0.00f, 0.30f,  0.00f },  /* head   */
    {  0.18f, 0.16f,  0.00f },  /* L upperarm */
    {  0.26f, 0.00f,  0.00f },  /* L forearm  */
    { -0.18f, 0.16f,  0.00f },  /* R upperarm */
    { -0.26f, 0.00f,  0.00f },  /* R forearm  */
    {  0.10f,-0.06f,  0.00f },  /* L thigh */
    {  0.00f,-0.42f,  0.00f },  /* L shin  */
    {  0.00f,-0.42f,  0.00f },  /* L foot  */
    { -0.10f,-0.06f,  0.00f },  /* R thigh */
    {  0.00f,-0.42f,  0.00f },  /* R shin  */
    {  0.00f,-0.42f,  0.00f },  /* R foot  */
};

/* ------------------------------------------------------------------ small math */

static void quat_axis(int axis, float angle, float q[4]) {
    float h = angle * 0.5f, s = sinf(h), c = cosf(h);
    q[0] = q[1] = q[2] = 0.0f; q[3] = c;
    q[axis] = s;
}

/* invert a rigid-ish 3x4 affine (row-major: [r0|t0, r1|t1, r2|t2]) into out (same layout). */
static void invert_affine(const float m[12], float out[12]) {
    float a = m[0], b = m[1], c = m[2];
    float d = m[4], e = m[5], f = m[6];
    float g = m[8], h = m[9], i = m[10];
    float A =  (e*i - f*h), B = -(d*i - f*g), C =  (d*h - e*g);
    float det = a*A + b*B + c*C;
    float inv = det != 0.0f ? 1.0f / det : 0.0f;
    /* inverse(M) = adjugate^T / det */
    float mi[9];
    mi[0] = A * inv;            mi[1] = -(b*i - c*h) * inv; mi[2] =  (b*f - c*e) * inv;
    mi[3] = B * inv;            mi[4] =  (a*i - c*g) * inv; mi[5] = -(a*f - c*d) * inv;
    mi[6] = C * inv;            mi[7] = -(a*h - b*g) * inv; mi[8] =  (a*e - b*d) * inv;
    float tx = m[3], ty = m[7], tz = m[11];
    out[0] = mi[0]; out[1] = mi[1]; out[2]  = mi[2];  out[3]  = -(mi[0]*tx + mi[1]*ty + mi[2]*tz);
    out[4] = mi[3]; out[5] = mi[4]; out[6]  = mi[5];  out[7]  = -(mi[3]*tx + mi[4]*ty + mi[5]*tz);
    out[8] = mi[6]; out[9] = mi[7]; out[10] = mi[8];  out[11] = -(mi[6]*tx + mi[7]*ty + mi[8]*tz);
}

/* ------------------------------------------------------------------ rest / bind pose */

/* Compose bind-pose model affines from rest locals, and derive inverse-bind. bind_pos[j] is the
 * joint origin in bind space (used for mesh placement). */
static void build_bind(float rest_local[J_COUNT * 10], float inverse_bind[J_COUNT * 12],
                       float bind_pos[J_COUNT][3]) {
    float model[J_COUNT][12];
    for (int j = 0; j < J_COUNT; ++j) {
        float *r = &rest_local[j * 10];
        r[0] = 0; r[1] = 0; r[2] = 0; r[3] = 1;                 /* q */
        r[4] = k_rest_t[j][0]; r[5] = k_rest_t[j][1]; r[6] = k_rest_t[j][2];
        r[7] = 1; r[8] = 1; r[9] = 1;                            /* s */

        mrw_trs trs;
        memcpy(trs.rot, &r[0], 4 * sizeof(float));
        memcpy(trs.trans, &r[4], 3 * sizeof(float));
        memcpy(trs.scale, &r[7], 3 * sizeof(float));
        float local[12];
        mrw_trs_to_affine(&trs, local);

        if (k_parent[j] == 0xFFFF) memcpy(model[j], local, sizeof local);
        else mrw_affine_mul(model[k_parent[j]], local, model[j]);

        invert_affine(model[j], &inverse_bind[j * 12]);
        bind_pos[j][0] = model[j][3]; bind_pos[j][1] = model[j][7]; bind_pos[j][2] = model[j][11];
    }
}

/* ------------------------------------------------------------------ clip animation */

/* Fill per-joint local quaternions for a pose at normalized phase ph in [0,1). Also returns a
 * vertical bob added to the pelvis local translation. kind: 0 idle, 1 walk, 2 run. */
static void anim_pose(int kind, float ph, float q[J_COUNT][4], float *pelvis_bob) {
    float p = ph * 6.2831853f;
    for (int j = 0; j < J_COUNT; ++j) { q[j][0] = q[j][1] = q[j][2] = 0.0f; q[j][3] = 1.0f; }
    *pelvis_bob = 0.0f;

    if (kind == 0) {                                   /* idle: subtle breathing/sway */
        quat_axis(0, 0.03f * sinf(p),        q[J_SPINE]);
        quat_axis(0, -0.02f * sinf(p),       q[J_CHEST]);
        quat_axis(1, 0.06f * sinf(0.5f * p), q[J_HEAD]);
        quat_axis(0, 0.05f * sinf(p),        q[J_LUARM]);
        quat_axis(0, 0.05f * sinf(p),        q[J_RUARM]);
        *pelvis_bob = 0.01f * sinf(p);
        return;
    }

    float amp_thigh = (kind == 2) ? 0.85f : 0.55f;
    float amp_knee  = (kind == 2) ? 1.10f : 0.75f;
    float amp_arm   = (kind == 2) ? 0.70f : 0.45f;
    float lean      = (kind == 2) ? 0.22f : 0.05f;

    quat_axis(0,  amp_thigh * sinf(p),                         q[J_LTHIGH]);
    quat_axis(0,  amp_thigh * sinf(p + 3.14159f),             q[J_RTHIGH]);
    quat_axis(0,  amp_knee  * (0.5f - 0.5f * cosf(p)),        q[J_LSHIN]);
    quat_axis(0,  amp_knee  * (0.5f - 0.5f * cosf(p + 3.14159f)), q[J_RSHIN]);
    quat_axis(0, -0.30f * sinf(p),                            q[J_LFOOT]);
    quat_axis(0, -0.30f * sinf(p + 3.14159f),                 q[J_RFOOT]);
    quat_axis(0, -amp_arm * sinf(p),                          q[J_LUARM]);
    quat_axis(0, -amp_arm * sinf(p + 3.14159f),               q[J_RUARM]);
    quat_axis(0,  0.30f * (0.5f - 0.5f * cosf(p + 3.14159f)), q[J_LLARM]);
    quat_axis(0,  0.30f * (0.5f - 0.5f * cosf(p)),            q[J_RLARM]);
    quat_axis(0,  lean,                                       q[J_SPINE]);
    quat_axis(1,  0.07f * sinf(p),                            q[J_CHEST]);
    *pelvis_bob = ((kind == 2) ? 0.04f : 0.025f) * cosf(2.0f * p);
}

/* Author one clip's dense joint-major samples (+ optional root track). Caller owns the buffers. */
static void author_clip(int kind, uint32_t sample_count, float *samples, float *root_track,
                        float duration, float fwd_speed) {
    for (uint32_t s = 0; s < sample_count; ++s) {
        float ph = (float)s / (float)(sample_count - 1);
        float q[J_COUNT][4]; float bob;
        anim_pose(kind, ph, q, &bob);
        for (int j = 0; j < J_COUNT; ++j) {
            float *o = &samples[((size_t)j * sample_count + s) * 10];
            o[0] = q[j][0]; o[1] = q[j][1]; o[2] = q[j][2]; o[3] = q[j][3];
            o[4] = k_rest_t[j][0]; o[5] = k_rest_t[j][1]; o[6] = k_rest_t[j][2];
            if (j == J_PELVIS) o[5] += bob;
            o[7] = 1.0f; o[8] = 1.0f; o[9] = 1.0f;
        }
        if (root_track) {
            float *r = &root_track[(size_t)s * 7];
            r[0] = 0; r[1] = 0; r[2] = 0; r[3] = 1;            /* no turning */
            r[4] = 0.0f;
            r[5] = 0.0f;
            r[6] = -fwd_speed * ph * duration;                /* forward = -Z */
        }
    }
}

/* ------------------------------------------------------------------ mesh (box per bone) */

typedef struct { DemoVertex *v; uint32_t *idx; uint32_t vc, ic; } MeshBuf;

static void emit_quad(MeshBuf *m, const float c[4][3], const float n[3], const float tan[3], uint32_t bone) {
    uint32_t b = m->vc;
    for (int k = 0; k < 4; ++k) {
        DemoVertex *d = &m->v[m->vc++];
        d->pos[0] = c[k][0]; d->pos[1] = c[k][1]; d->pos[2] = c[k][2];
        d->nrm[0] = n[0]; d->nrm[1] = n[1]; d->nrm[2] = n[2];
        d->tan[0] = tan[0]; d->tan[1] = tan[1]; d->tan[2] = tan[2]; d->tan[3] = 1.0f;
        d->bones[0] = bone; d->bones[1] = 0; d->bones[2] = 0; d->bones[3] = 0;
        d->weights[0] = 1.0f; d->weights[1] = d->weights[2] = d->weights[3] = 0.0f;
    }
    m->idx[m->ic++] = b + 0; m->idx[m->ic++] = b + 1; m->idx[m->ic++] = b + 2;
    m->idx[m->ic++] = b + 0; m->idx[m->ic++] = b + 2; m->idx[m->ic++] = b + 3;
}

/* (u, v, axis) is right-handed, so each face's corners are ordered so emit_quad's CCW triangles
 * (0,1,2)+(0,2,3) wind toward the OUTWARD normal - required for back-face culling. */
static void emit_box(MeshBuf *m, const float C[3], const float axis[3], float hl,
                     const float u[3], const float v[3], float r, uint32_t bone) {
    /* corner(se,su,sv) */
    #define CORNER(se, su, sv, out) do { \
        (out)[0] = C[0] + (se)*hl*axis[0] + (su)*r*u[0] + (sv)*r*v[0]; \
        (out)[1] = C[1] + (se)*hl*axis[1] + (su)*r*u[1] + (sv)*r*v[1]; \
        (out)[2] = C[2] + (se)*hl*axis[2] + (su)*r*u[2] + (sv)*r*v[2]; } while (0)
    float q[4][3];
    /* +axis */
    CORNER( 1,-1,-1,q[0]); CORNER( 1, 1,-1,q[1]); CORNER( 1, 1, 1,q[2]); CORNER( 1,-1, 1,q[3]);
    emit_quad(m, q, axis, u, bone);
    /* -axis */
    { float na[3] = { -axis[0], -axis[1], -axis[2] };
      CORNER(-1,-1,-1,q[0]); CORNER(-1,-1, 1,q[1]); CORNER(-1, 1, 1,q[2]); CORNER(-1, 1,-1,q[3]);
      emit_quad(m, q, na, u, bone); }
    /* +u  (corners reversed vs the others so CCW winds toward +u, not -u) */
    CORNER(-1, 1, 1,q[0]); CORNER( 1, 1, 1,q[1]); CORNER( 1, 1,-1,q[2]); CORNER(-1, 1,-1,q[3]);
    emit_quad(m, q, u, axis, bone);
    /* -u */
    { float nu[3] = { -u[0], -u[1], -u[2] };
      CORNER( 1,-1,-1,q[0]); CORNER( 1,-1, 1,q[1]); CORNER(-1,-1, 1,q[2]); CORNER(-1,-1,-1,q[3]);
      emit_quad(m, q, nu, axis, bone); }
    /* +v */
    CORNER(-1,-1, 1,q[0]); CORNER( 1,-1, 1,q[1]); CORNER( 1, 1, 1,q[2]); CORNER(-1, 1, 1,q[3]);
    emit_quad(m, q, v, axis, bone);
    /* -v */
    { float nv[3] = { -v[0], -v[1], -v[2] };
      CORNER(-1,-1,-1,q[0]); CORNER(-1, 1,-1,q[1]); CORNER( 1, 1,-1,q[2]); CORNER( 1,-1,-1,q[3]);
      emit_quad(m, q, nv, axis, bone); }
    #undef CORNER
}

static void basis_from_axis(const float axis[3], float u[3], float v[3]) {
    float up[3] = { 0, 1, 0 };
    if (fabsf(axis[1]) > 0.99f) { up[0] = 1; up[1] = 0; up[2] = 0; }
    /* u = normalize(cross(up, axis)) */
    u[0] = up[1]*axis[2] - up[2]*axis[1];
    u[1] = up[2]*axis[0] - up[0]*axis[2];
    u[2] = up[0]*axis[1] - up[1]*axis[0];
    float ul = sqrtf(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    if (ul > 0) { u[0] /= ul; u[1] /= ul; u[2] /= ul; }
    /* v = cross(axis, u) */
    v[0] = axis[1]*u[2] - axis[2]*u[1];
    v[1] = axis[2]*u[0] - axis[0]*u[2];
    v[2] = axis[0]*u[1] - axis[1]*u[0];
}

/* radius (half-thickness) per bone for the box mesh */
static float bone_radius(int j) {
    switch (j) {
        case J_PELVIS: case J_SPINE: case J_CHEST: return 0.12f;
        case J_HEAD: return 0.11f;
        case J_LUARM: case J_RUARM: case J_LLARM: case J_RLARM: return 0.045f;
        default: return 0.07f;   /* legs */
    }
}

static void build_mesh(float bind_pos[J_COUNT][3], MeshBuf *m) {
    int child_count[J_COUNT] = { 0 };
    for (int j = 0; j < J_COUNT; ++j)
        if (k_parent[j] != 0xFFFF) child_count[k_parent[j]]++;

    uint32_t boxes = 0;
    for (int j = 0; j < J_COUNT; ++j) boxes += child_count[j] > 0 ? (uint32_t)child_count[j] : 1u;

    m->v = (DemoVertex *)malloc((size_t)boxes * 24 * sizeof(DemoVertex));
    m->idx = (uint32_t *)malloc((size_t)boxes * 36 * sizeof(uint32_t));
    m->vc = 0; m->ic = 0;

    for (int j = 0; j < J_COUNT; ++j) {
        float r = bone_radius(j);
        if (child_count[j] == 0) {
            /* leaf: small cube at the joint, weighted to itself */
            float axis[3] = { 0, 1, 0 }, u[3], v[3];
            basis_from_axis(axis, u, v);
            emit_box(m, bind_pos[j], axis, r, u, v, r, (uint32_t)j);
        } else {
            for (int c = 0; c < J_COUNT; ++c) {
                if (k_parent[c] != (uint16_t)j) continue;
                float A[3] = { bind_pos[j][0], bind_pos[j][1], bind_pos[j][2] };
                float B[3] = { bind_pos[c][0], bind_pos[c][1], bind_pos[c][2] };
                float d[3] = { B[0]-A[0], B[1]-A[1], B[2]-A[2] };
                float len = sqrtf(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
                if (len < 1e-5f) len = 2.0f * r;
                float axis[3] = { d[0]/len, d[1]/len, d[2]/len }, u[3], v[3];
                basis_from_axis(axis, u, v);
                float C[3] = { (A[0]+B[0])*0.5f, (A[1]+B[1])*0.5f, (A[2]+B[2])*0.5f };
                emit_box(m, C, axis, len * 0.5f, u, v, r, (uint32_t)j);  /* segment weighted to proximal joint */
            }
        }
    }
}

/* ------------------------------------------------------------------ build everything */

static void *alloc16(size_t n) { return mrw_authoring_alloc(n); }

int assets_proc_build(ProcAssets *out) {
    memset(out, 0, sizeof *out);
    out->joint_count = J_COUNT;

    /* skeleton arrays */
    static float rest_local[J_COUNT * 10];
    static float inverse_bind[J_COUNT * 12];
    float bind_pos[J_COUNT][3];
    build_bind(rest_local, inverse_bind, bind_pos);

    mrw_skel skel = { J_COUNT, k_parent, rest_local, inverse_bind, k_names };

    /* clips: idle, walk, run */
    struct { int kind; float dur, fps, speed; int loop, root; } spec[] = {
        { 0, 1.6f, 30.0f, 0.0f, 1, 0 },   /* idle */
        { 1, 1.0f, 30.0f, 1.4f, 1, 1 },   /* walk */
        { 2, 0.6f, 30.0f, 3.6f, 1, 1 },   /* run  */
    };
    uint32_t nclip = 3;
    out->clip_count = nclip;

    mrw_clip clips[DEMO_PROC_MAX_CLIPS] = { 0 };   /* zero so `codec` defaults to 0 (raw TRS) - the
                                                      loop below sets every field except codec, and an
                                                      uninitialized codec > 1 is rejected as UNSUPPORTED */
    float *samples_buf[DEMO_PROC_MAX_CLIPS] = { 0 };
    float *root_buf[DEMO_PROC_MAX_CLIPS] = { 0 };
    for (uint32_t i = 0; i < nclip; ++i) {
        uint32_t sc = (uint32_t)(spec[i].dur * spec[i].fps + 0.5f) + 1;
        if (sc < 2) sc = 2;
        samples_buf[i] = (float *)malloc((size_t)J_COUNT * sc * 10 * sizeof(float));
        root_buf[i] = spec[i].root ? (float *)malloc((size_t)sc * 7 * sizeof(float)) : NULL;
        author_clip(spec[i].kind, sc, samples_buf[i], root_buf[i], spec[i].dur, spec[i].speed);
        clips[i].fps = spec[i].fps;
        clips[i].sample_count = sc;
        clips[i].flags = (spec[i].loop ? MRW_CLIP_LOOPING : 0u) |
                         (spec[i].root ? MRW_CLIP_HAS_ROOT_MOTION : 0u);
        clips[i].samples = samples_buf[i];
        clips[i].root_track = root_buf[i];

        out->clips[i].name = (spec[i].kind == 0) ? "idle" : (spec[i].kind == 1) ? "walk" : "run";
        out->clips[i].clip_index = i;
        out->clips[i].duration_s = (float)(sc - 1) / spec[i].fps;
        out->clips[i].looping = spec[i].loop;
        out->clips[i].has_root_motion = spec[i].root;
    }

    /* 1) CPU blob, 2) open + bake each clip, 3) re-emit with the BAKED section. */
    uint8_t *tierA = NULL; size_t tierA_size = 0;
    uint16_t *combined = NULL;   /* declared here so the cleanup labels can free it */
    if (mrw_authoring_build(&skel, clips, nclip, NULL, &tierA, &tierA_size) != MRW_OK) {
        fprintf(stderr, "[assets] Tier-A authoring failed\n"); goto fail;
    }
    mrw_blob blob;
    if (mrw_blob_open(tierA, tierA_size, &blob) != MRW_OK) { fprintf(stderr, "[assets] blob open failed\n"); goto fail_tierA; }
    mrw_skeleton_view sv;
    mrw_blob_skeleton(&blob, &sv);

    const float bake_fps = 24.0f;
    uint32_t total_frames = 0;
    uint32_t frame_counts[DEMO_PROC_MAX_CLIPS];
    for (uint32_t i = 0; i < nclip; ++i) {
        uint32_t fc = (uint32_t)(out->clips[i].duration_s * bake_fps + 0.5f) + 1;
        if (fc < 2) fc = 2;
        frame_counts[i] = fc;
        total_frames += fc;
    }
    uint32_t stride = J_COUNT * 2;
    combined = (uint16_t *)malloc((size_t)total_frames * stride * 4 * sizeof(uint16_t));

    uint32_t bidx[DEMO_PROC_MAX_CLIPS], bff[DEMO_PROC_MAX_CLIPS], bfc[DEMO_PROC_MAX_CLIPS], bflags[DEMO_PROC_MAX_CLIPS];
    float bdur[DEMO_PROC_MAX_CLIPS];
    int all_eligible = 1;
    uint32_t cursor = 0;
    for (uint32_t i = 0; i < nclip; ++i) {
        mrw_clip_view cv;
        if (mrw_clip_view_at(&blob, 1 + i, &cv) != MRW_OK) { fprintf(stderr, "[assets] clip view %u failed\n", i); goto fail_combined; }
        mrw_mem_req sreq, treq;
        mrw_bake_clip_requirements(J_COUNT, frame_counts[i], &sreq, &treq);
        void *scratch = alloc16(sreq.size);
        uint16_t *texels = (uint16_t *)alloc16(treq.size);
        mrw_bake_stats stats;
        mrw_result br = mrw_bake_clip(&sv, &cv, frame_counts[i], NULL, NULL, 0.0f,
                                      scratch, sreq.size, texels, treq.size, &stats);
        if (br != MRW_OK) { fprintf(stderr, "[assets] bake clip %u error\n", i); mrw_authoring_free(scratch); mrw_authoring_free(texels); goto fail_combined; }
        if (!stats.eligible) all_eligible = 0;
        fprintf(stderr, "[assets] bake '%s': %u frames, %s (max residual %.4g mm)\n",
                out->clips[i].name, frame_counts[i], stats.eligible ? "eligible" : "INELIGIBLE",
                (double)stats.max_residual * 1000.0);
        memcpy(&combined[(size_t)cursor * stride * 4], texels, (size_t)frame_counts[i] * stride * 4 * sizeof(uint16_t));
        bidx[i] = i; bff[i] = cursor; bfc[i] = frame_counts[i];
        bdur[i] = out->clips[i].duration_s;
        bflags[i] = out->clips[i].looping ? MRW_BAKED_CLIP_LOOPING : 0u;
        cursor += frame_counts[i];
        mrw_authoring_free(scratch); mrw_authoring_free(texels);
    }
    out->tier_b_eligible = all_eligible;

    mrw_baked baked = { 0, total_frames, combined, nclip, bidx, bff, bfc, bdur, bflags };
    if (mrw_authoring_build(&skel, clips, nclip, &baked, &out->blob, &out->blob_size) != MRW_OK) {
        fprintf(stderr, "[assets] final authoring failed\n"); goto fail_combined;
    }
    /* self-validate the final blob */
    mrw_blob check;
    if (mrw_blob_open(out->blob, out->blob_size, &check) != MRW_OK) {
        fprintf(stderr, "[assets] final blob failed validation\n"); goto fail_combined;
    }

    free(combined);
    mrw_authoring_free(tierA);
    for (uint32_t i = 0; i < nclip; ++i) { free(samples_buf[i]); free(root_buf[i]); }

    /* mesh */
    MeshBuf mesh;
    build_mesh(bind_pos, &mesh);
    out->verts = mesh.v; out->vert_count = mesh.vc;
    out->indices = mesh.idx; out->index_count = mesh.ic;
    out->mesh_cull_back = 1;   /* emit_box winds CCW-outward on every face (see winding note) */

    fprintf(stderr, "[assets] biped: %u joints, %u clips, %u baked frames, mesh %u verts / %u tris%s\n",
            (unsigned)J_COUNT, nclip, total_frames, mesh.vc, mesh.ic / 3,
            all_eligible ? "" : "  (some clips INELIGIBLE for Tier B)");
    return 0;

fail_combined:
    free(combined);
fail_tierA:
    mrw_authoring_free(tierA);
fail:
    for (uint32_t i = 0; i < nclip; ++i) { free(samples_buf[i]); free(root_buf[i]); }
    return 1;
}

void assets_proc_free(ProcAssets *out) {
    if (out->blob) mrw_authoring_free(out->blob);
    free(out->verts);
    free(out->indices);
    memset(out, 0, sizeof *out);
}
