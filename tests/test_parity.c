/* CPU-tier ↔ baked-tier parity. Proves the *honest*
 * relationship: the baked tier is EXACT at baked frames (within binary16), APPROXIMATE between them,
 * and a promotion baked→CPU pops by a bounded, measured amount. Driven by the prototype baker
 * (tests/mrw_bake.{h,c}); the runtime stays consume-only (mrw_baked_sample_bone). Sections:
 *   A  mat3→quat edge cases (independent of the bake roundtrip)
 *   B  decompose hard checks + the eligibility metric (rigid/uniform exact, shear/singular rejected)
 *   C  exact-frame parity + MEASURED half-float error; rigid ⇒ decompose-exact; clip_id keying
 *   D  between-frame chord-vs-arc error, shrinking with bake density
 *   E  looping endpoint behaviour (t=dur wrap, dur−ε, negative t) + runtime-vs-component reference
 *   F  promotion-pop at arbitrary phase + direct baked quaternion sign-continuity
 *   G  ≤2-clip cross-fade gap, interpolated in Q/T/s component space
 *   H  whole-rig eligibility: a non-uniform-scale bone rejects the entire rig */
#include "test_util.h"
#include "mrw_build.h"
#include "mrw_bake.h"
#include "marrow_internal.h"   /* mrw_quat_nlerp - shared nlerp for the CPU-tier blend reference */
#include <math.h>
#include <stdio.h>
#include <string.h>

#define FPS   30.0f
#define MAXJ  16u
#define MAXS  40u
#define MAXF  96u
#define NP    8u   /* probes/bone: 8 corners of a small bind-space box (influence-AABB stand-in) */

/* ---- deterministic rig generation (local copy so parent/rest stay visible for probes) ---- */
typedef struct { uint32_t s; } rng;
static uint32_t ru32(rng *r) { r->s = r->s*1664525u + 1013904223u; return r->s; }
static float    rf(rng *r, float lo, float hi) { return lo + (hi-lo)*((float)(ru32(r)>>8)/(float)(1u<<24)); }
static void rquat(rng *r, float q[4], float range) {
    float ax=rf(r,-1,1), ay=rf(r,-1,1), az=rf(r,-1,1);
    float n=sqrtf(ax*ax+ay*ay+az*az); if (n<1e-6f){ax=0;ay=0;az=1;n=1;}
    float ang=rf(r,-range,range), s=sinf(ang*0.5f);
    q[0]=ax/n*s; q[1]=ay/n*s; q[2]=az/n*s; q[3]=cosf(ang*0.5f);
}

/* Fill a SMOOTH, coherent clip: each joint swings about a fixed axis with a sinusoidal angle
 * (amplitude `amp`) plus a small sinusoidal translation. Real animation is band-limited like
 * this - that's the premise of "bake low, interpolate in-shader": denser baking then reduces
 * the chord-vs-arc gap. (White-noise per-sample rotation would be unbakeable by construction.) */
static void fill_smooth_clip(uint32_t jc, uint32_t sc, uint32_t seed, float amp, float *samples) {
    rng r = { seed ? seed : 1u };
    for (uint32_t j = 0; j < jc; ++j) {
        float ax=rf(&r,-1,1), ay=rf(&r,-1,1), az=rf(&r,-1,1);
        float n=sqrtf(ax*ax+ay*ay+az*az); if (n<1e-6f){ax=0;ay=0;az=1;n=1;} ax/=n; ay/=n; az/=n;
        float freq=rf(&r,0.3f,0.8f), ph=rf(&r,0.0f,6.2831853f);
        float bx=rf(&r,-0.15f,0.15f), by=rf(&r,-0.15f,0.15f), bz=rf(&r,-0.15f,0.15f);
        float tamp=rf(&r,0.0f,0.12f), tph=rf(&r,0.0f,6.2831853f);
        for (uint32_t s = 0; s < sc; ++s) {
            float u = (sc>1) ? (float)s/(float)(sc-1) : 0.0f;
            float ang = amp * sinf(6.2831853f*freq*u + ph);
            float sh = sinf(ang*0.5f), ch = cosf(ang*0.5f);
            float tw = sinf(6.2831853f*freq*u + tph);
            float *smp = samples + ((size_t)j*sc + s)*10;
            smp[0]=ax*sh; smp[1]=ay*sh; smp[2]=az*sh; smp[3]=ch;
            smp[4]=bx+tamp*tw; smp[5]=by+tamp*tw; smp[6]=bz+tamp*tw;
            smp[7]=smp[8]=smp[9]=1.0f;     /* unit scale ⇒ rigid clip locals */
        }
    }
}

/* Build a human-scale chain rig (reach ~2-3 m) + a smooth clip. `nonuniform_bone` (or -1):
 * give that bone a non-uniform REST scale so its skinning transform stops being a similarity
 * (baked-tier-ineligible). `amp` is the clip's per-joint rotation amplitude (radians). */
static void gen_rig(uint32_t jc, uint32_t sc, uint32_t seed, int nonuniform_bone, float amp,
                    uint16_t *parent, float *rest, float *ib, float *samples) {
    rng r = { seed ? seed : 1u };
    for (uint32_t j = 0; j < jc; ++j) {
        parent[j] = (j==0) ? 0xFFFFu : (uint16_t)(j-1);
        float q[4]; rquat(&r, q, 3.14159265f);
        rest[j*10+0]=q[0]; rest[j*10+1]=q[1]; rest[j*10+2]=q[2]; rest[j*10+3]=q[3];
        rest[j*10+4]=(j==0)?0.0f:rf(&r,0.12f,0.28f);   /* child offset along the chain */
        rest[j*10+5]=rf(&r,-0.1f,0.1f);
        rest[j*10+6]=rf(&r,-0.1f,0.1f);
        rest[j*10+7]=rest[j*10+8]=rest[j*10+9]=1.0f;
        if ((int)j == nonuniform_bone) { rest[j*10+7]=2.0f; rest[j*10+8]=0.5f; rest[j*10+9]=1.3f; }
    }
    compute_bind_inverse(jc, parent, rest, ib);
    fill_smooth_clip(jc, sc, seed ^ 0x9e3779b9u, amp, samples);
}

/* Per-bone bind-space probes: 8 corners of a [-r,r]³ box around the bone's bind position. */
static void make_probes(uint32_t jc, const uint16_t *parent, const float *rest, float *out) {
    static float model[MAXJ*12];
    for (uint32_t j = 0; j < jc; ++j) {
        mrw_trs trs;
        memcpy(trs.rot, rest+j*10+0, 16);
        memcpy(trs.trans, rest+j*10+4, 12);
        memcpy(trs.scale, rest+j*10+7, 12);
        float local[12]; mrw_trs_to_affine(&trs, local);
        if (parent[j]==0xFFFF) memcpy(model+j*12, local, sizeof local);
        else mrw_affine_mul(model+(size_t)parent[j]*12, local, model+j*12);
    }
    const float rr = 0.15f;
    for (uint32_t j = 0; j < jc; ++j) {
        float bx=model[j*12+3], by=model[j*12+7], bz=model[j*12+11];
        uint32_t k=0;
        for (int sx=-1; sx<=1; sx+=2)
        for (int sy=-1; sy<=1; sy+=2)
        for (int sz=-1; sz<=1; sz+=2) {
            float *p = out + ((size_t)j*NP + k)*3;
            p[0]=bx+sx*rr; p[1]=by+sy*rr; p[2]=bz+sz*rr; ++k;
        }
    }
}

static const char *NAMES[MAXJ] = {
    "j0","j1","j2","j3","j4","j5","j6","j7","j8","j9","j10","j11","j12","j13","j14","j15"
};

/* ---- shared scratch ---- */
static uint16_t g_parent[MAXJ];
static float    g_rest[MAXJ*10], g_ib[MAXJ*12], g_samp[MAXJ*MAXS*10], g_samp2[MAXJ*MAXS*10];
static float    g_probes[MAXJ*NP*3];
static _Alignas(16) uint16_t g_tex[MAXF*MAXJ*2*4];   /* bake core needs a 16-aligned texel buffer */
static float    g_model[MAXJ*12], g_palA[MAXJ*12];
/* caller-owned bake scratch (model+palette+prevq = 28 floats/joint) + per-bone probe counts. */
static _Alignas(16) float g_bake_scratch[MAXJ*(12+12+4)];
static uint32_t g_pcounts[MAXJ];

/* Open a {skel, clips[, baked]} blob (asserts validity). Free *buf with mrw_free. */
static uint8_t *open_blob(const mrw_skel *skel, const mrw_clip *clips, uint32_t nclip,
                          const mrw_baked *baked, mrw_blob *blob) {
    uint8_t *buf=NULL; size_t sz=mrw_build(skel, clips, nclip, baked, &buf);
    CHECK_EQ(mrw_blob_open(buf, sz, blob), MRW_OK);
    return buf;
}

/* Bake clip `ci` of a built blob at `fc` frames into `tex` (writing frames starting at
 * absolute `first_frame`); returns the whole-rig eligibility verdict. */
static int bake_into(const mrw_blob *blob, uint32_t ci, uint32_t fc, uint32_t first_frame,
                     uint16_t *tex, float tol, float *resid) {
    mrw_skeleton_view sv; mrw_clip_view cv;
    CHECK_EQ(mrw_blob_skeleton(blob, &sv), MRW_OK);
    CHECK_EQ(mrw_clip_view_at(blob, ci, &cv), MRW_OK);
    uint32_t jc = sv.joint_count;
    uint32_t fst = jc * 2;
    uint16_t *base = tex + (size_t)first_frame * fst * 4;   /* 16-aligned: offset is jc·16·first_frame bytes */
    for (uint32_t j = 0; j < jc; ++j) g_pcounts[j] = NP;    /* uniform NP probes/bone (flat layout == g_probes) */
    mrw_mem_req sreq, treq;
    CHECK_EQ(mrw_bake_clip_requirements(jc, fc, &sreq, &treq), MRW_OK);
    mrw_bake_stats st;
    CHECK_EQ(mrw_bake_clip(&sv, &cv, fc, g_pcounts, g_probes, tol,
                           g_bake_scratch, sizeof g_bake_scratch, base, treq.size, &st), MRW_OK);
    if (resid) *resid = st.max_residual;
    return st.eligible;
}

/* worst CPU-tier-vs-baked-tier probe displacement over the whole skeleton at clip-local time t. */
static float tierAB_pop(const mrw_skeleton_view *sv, const mrw_clip_view *cv,
                        const mrw_baked_view *bv, uint32_t clip_index, float t, const float *probes) {
    mrw_clip_to_palette(sv, cv, t, g_model, g_palA, sv->joint_count);
    float worst = 0.0f;
    for (uint32_t b = 0; b < sv->joint_count; ++b) {
        float affB[12];
        mrw_baked_sample_bone(bv, clip_index, b, t, affB);
        float d = mrw_affine_probe_dist(g_palA+(size_t)b*12, affB, NP, probes+(size_t)b*NP*3);
        if (d > worst) worst = d;
    }
    return worst;
}

/* ===================================================================== A: mat3→quat */
static void test_mat3_to_quat(void) {
    /* unit quats exercising every Shepperd branch (trace>0, and each largest-diagonal case
     * via ~180° rotations), compared through the reconstructed matrix so the q/−q sign
     * ambiguity is irrelevant. */
    float qs[][4] = {
        {0,0,0,1},                                  /* identity (trace>0)        */
        {0.70710678f,0,0,0.70710678f},              /* 90° x (trace>0)           */
        {0,0.70710678f,0,0.70710678f},              /* 90° y                     */
        {0,0,0.70710678f,0.70710678f},              /* 90° z                     */
        {1,0,0,0}, {0,1,0,0}, {0,0,1,0},            /* 180° x/y/z (largest-diag) */
        {0.9998477f,0.0174524f,0,0.0f},             /* ~178° about ~x (trace<0)  */
        {0.3320391f,0.4428521f,0.6640782f,0.5f},    /* arbitrary axis/angle      */
    };
    int n = (int)(sizeof qs / sizeof qs[0]);
    float worst = 0.0f;
    for (int i = 0; i < n; ++i) {
        float nrm = sqrtf(qs[i][0]*qs[i][0]+qs[i][1]*qs[i][1]+qs[i][2]*qs[i][2]+qs[i][3]*qs[i][3]);
        for (int k=0;k<4;k++) qs[i][k] /= nrm;
        float R[9];  mrw_quat_to_mat3(qs[i], R);
        float q2[4]; mrw_mat3_to_quat(R, q2);
        float R2[9]; mrw_quat_to_mat3(q2, R2);
        for (int k=0;k<9;k++){ float d=fabsf(R[k]-R2[k]); if(d>worst)worst=d; CHECK_NEAR(R2[k], R[k], 1e-5); }
        float n2=q2[0]*q2[0]+q2[1]*q2[1]+q2[2]*q2[2]+q2[3]*q2[3];
        CHECK_NEAR(n2, 1.0f, 1e-5);
    }
    printf("  [A] mat3->quat: %d cases, worst |R-R'| = %.2e\n", n, worst);
}

/* ===================================================================== B: decompose checks */
static void test_decompose_checks(void) {
    /* origin-offset box probes so a shear (which fixes the origin) shows nonzero residual. */
    float probes[NP*3]; uint32_t k=0;
    for (int sx=-1;sx<=1;sx+=2) for (int sy=-1;sy<=1;sy+=2) for (int sz=-1;sz<=1;sz+=2) {
        probes[k*3+0]=0.5f*sx; probes[k*3+1]=0.5f*sy; probes[k*3+2]=0.5f*sz; ++k;
    }
    float q[4]={0.18257419f,0.36514837f,0.54772256f,0.73029674f}; /* arbitrary unit */
    float R[9]; mrw_quat_to_mat3(q, R);

    /* pure rotation ⇒ eligible, residual ≈ 0, scale ≈ 1 */
    float rot12[12]={R[0],R[1],R[2],0.3f, R[3],R[4],R[5],-0.2f, R[6],R[7],R[8],0.1f};
    mrw_xform xr; CHECK_EQ(mrw_decompose_affine(rot12,&xr), 1);
    CHECK_NEAR(xr.scale, 1.0f, 1e-4);
    CHECK(mrw_decompose_residual(rot12, NP, probes) < 1e-4);

    /* uniform scale 2 ⇒ eligible, scale ≈ 2, residual ≈ 0 */
    float us12[12]; for(int i=0;i<12;i++) us12[i]=rot12[i];
    for(int r=0;r<3;r++) for(int c=0;c<3;c++) us12[r*4+c]*=2.0f;
    mrw_xform xu; CHECK_EQ(mrw_decompose_affine(us12,&xu), 1);
    CHECK_NEAR(xu.scale, 2.0f, 1e-3);
    CHECK(mrw_decompose_residual(us12, NP, probes) < 1e-3);

    /* shear ⇒ structurally decomposes (det 1 > 0) but residual ≫ tol (not a similarity) */
    float sh12[12]={1,0.5f,0,0, 0,1,0,0, 0,0,1,0};
    mrw_xform xs; CHECK_EQ(mrw_decompose_affine(sh12,&xs), 1);
    float sh_res = mrw_decompose_residual(sh12, NP, probes);
    CHECK(sh_res > 1e-2);

    /* singular (zero z-axis) ⇒ ineligible; residual is +inf */
    float sg12[12]={1,0,0,0, 0,1,0,0, 0,0,0,0};
    mrw_xform xsg; CHECK_EQ(mrw_decompose_affine(sg12,&xsg), 0);
    CHECK(mrw_decompose_residual(sg12, NP, probes) > 1e30f);

    /* reflection (det < 0) ⇒ ineligible */
    float rfl[12]={R[0],R[1],R[2],0, R[3],R[4],R[5],0, -R[6],-R[7],-R[8],0};
    mrw_xform xrf; CHECK_EQ(mrw_decompose_affine(rfl,&xrf), 0);

    printf("  [B] decompose: rot/uniform residual<1e-3, shear residual=%.3e, singular/reflection rejected\n", sh_res);
}

/* ===================================================================== C: exact-frame + clip_id */
static void test_exact_frame(void) {
    uint32_t jc=8, sc=6;
    gen_rig(jc, sc, 0x11, -1, 0.8f, g_parent, g_rest, g_ib, g_samp);
    make_probes(jc, g_parent, g_rest, g_probes);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { FPS, sc, 0, g_samp, NULL };
    float dur = (float)(sc-1)/FPS;
    uint32_t fc = sc;     /* bake at clip fps ⇒ each baked frame lands on a keyframe */

    mrw_blob tmp; uint8_t *btmp = open_blob(&skel, &clip, 1, NULL, &tmp);
    float worst_resid = 0.0f;
    int eligible = bake_into(&tmp, 1, fc, 0, g_tex, 1e-3f, &worst_resid);
    mrw_free(btmp);
    CHECK_EQ(eligible, 1);                 /* rigid rig ⇒ baked-tier eligible */
    CHECK(worst_resid < 1e-4);             /* rigid ⇒ decomposition is exact */

    uint32_t bidx[1]={0}, bff[1]={0}, bfc[1]={fc}, bflags[1]={0}; float bdur[1]={dur};
    mrw_baked baked = { 0, fc, g_tex, 1, bidx, bff, bfc, bdur, bflags };
    mrw_blob blob; uint8_t *buf = open_blob(&skel, &clip, 1, &baked, &blob);
    mrw_skeleton_view sv; mrw_clip_view cv; mrw_baked_view bv;
    CHECK_EQ(mrw_blob_skeleton(&blob,&sv), MRW_OK);
    CHECK_EQ(mrw_clip_view_at(&blob,1,&cv), MRW_OK);
    CHECK_EQ(mrw_baked_view_at(&blob,2,&bv), MRW_OK);

    double sum2=0; uint32_t cnt=0; float maxe=0;
    for (uint32_t f=0; f<fc; ++f) {
        float t_f = (float)f/(float)(fc-1) * dur;
        mrw_clip_to_palette(&sv,&cv,t_f, g_model, g_palA, jc);
        for (uint32_t b=0;b<jc;++b){
            float affB[12]; mrw_baked_sample_bone(&bv, 0, b, t_f, affB);
            float e = mrw_affine_probe_dist(g_palA+(size_t)b*12, affB, NP, g_probes+(size_t)b*NP*3);
            sum2 += (double)e*e; ++cnt; if (e>maxe) maxe=e;
        }
    }
    float rms = (float)sqrt(sum2/(cnt?cnt:1));
    CHECK(maxe < 1.5e-3);     /* exact within binary16 - sub-2mm at this rig scale (measured ~0.6mm) */
    printf("  [C] exact-frame half-float error: max=%.3e m  rms=%.3e m  (%u samples)\n", maxe, rms, cnt);

    /* clip_id keying: the baked entry's clip_id resolves to the paired CLIP. */
    mrw_baked_clip ce; CHECK_EQ(mrw_baked_clip_entry(&bv, 0, &ce), MRW_OK);
    CHECK(mrw_id_equal(&ce.clip_id, &cv.id));
    mrw_clip_view cv2; CHECK_EQ(mrw_blob_clip_by_id(&blob, &ce.clip_id, &cv2), MRW_OK);
    CHECK(mrw_id_equal(&cv2.id, &cv.id));
    mrw_free(buf);
}

/* measure the worst off-grid CPU-tier-vs-baked-tier error for a clip baked at `fc` frames */
static float between_error_at_density(const mrw_skel *skel, const mrw_clip *clip,
                                      uint32_t jc, uint32_t sc, uint32_t fc) {
    float dur = (float)(sc-1)/FPS;
    mrw_blob tmp; uint8_t *btmp = open_blob(skel, clip, 1, NULL, &tmp);
    float resid; bake_into(&tmp, 1, fc, 0, g_tex, 1e9f, &resid); /* tol huge: density study only */
    mrw_free(btmp);
    uint32_t bidx[1]={0}, bff[1]={0}, bfc[1]={fc}, bflags[1]={0}; float bdur[1]={dur};
    mrw_baked baked = { 0, fc, g_tex, 1, bidx, bff, bfc, bdur, bflags };
    mrw_blob blob; uint8_t *buf = open_blob(skel, clip, 1, &baked, &blob);
    mrw_skeleton_view sv; mrw_clip_view cv; mrw_baked_view bv;
    mrw_blob_skeleton(&blob,&sv); mrw_clip_view_at(&blob,1,&cv); mrw_baked_view_at(&blob,2,&bv);
    float worst=0.0f; int steps=53;
    for (int i=1;i<steps;i++){
        float t = dur*(float)i/(float)steps;       /* strictly off the baked grid */
        float p = tierAB_pop(&sv,&cv,&bv,0,t,g_probes);
        if (p>worst) worst=p;
    }
    (void)jc;
    mrw_free(buf);
    return worst;
}

/* ===================================================================== D: between-frame density */
static void test_between_frame(void) {
    uint32_t jc=8, sc=17;
    gen_rig(jc, sc, 0x22, -1, 0.5f, g_parent, g_rest, g_ib, g_samp); /* smooth motion */
    make_probes(jc, g_parent, g_rest, g_probes);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { FPS, sc, 0, g_samp, NULL };

    float e_low  = between_error_at_density(&skel, &clip, jc, sc, 5);  /* 4 intervals */
    float e_high = between_error_at_density(&skel, &clip, jc, sc, 9);  /* 8 intervals */
    CHECK(e_low  < 0.5f);                 /* bounded                                       */
    CHECK(e_low  > 2.5f * e_high);        /* halving Δt cuts the chord gap super-linearly  */
    printf("  [D] between-frame chord-vs-arc: fc=5 max=%.3e m  fc=9 max=%.3e m  (ratio %.2f, ~O(Δt²))\n",
           e_low, e_high, e_low/(e_high>0?e_high:1));
}

/* ===================================================================== E: looping endpoints */
static void test_looping(void) {
    uint32_t jc=6, sc=8;
    gen_rig(jc, sc, 0x33, -1, 0.5f, g_parent, g_rest, g_ib, g_samp);
    make_probes(jc, g_parent, g_rest, g_probes);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { FPS, sc, MRW_CLIP_LOOPING, g_samp, NULL };
    float dur = (float)(sc-1)/FPS;
    uint32_t fc = sc;

    mrw_blob tmp; uint8_t *btmp = open_blob(&skel, &clip, 1, NULL, &tmp);
    float resid; CHECK_EQ(bake_into(&tmp, 1, fc, 0, g_tex, 1e-3f, &resid), 1);
    mrw_free(btmp);
    uint32_t bidx[1]={0}, bff[1]={0}, bfc[1]={fc}, bflags[1]={MRW_BAKED_CLIP_LOOPING}; float bdur[1]={dur};
    mrw_baked baked = { 0, fc, g_tex, 1, bidx, bff, bfc, bdur, bflags };
    mrw_blob blob; uint8_t *buf = open_blob(&skel, &clip, 1, &baked, &blob);
    mrw_skeleton_view sv; mrw_clip_view cv; mrw_baked_view bv;
    mrw_blob_skeleton(&blob,&sv); mrw_clip_view_at(&blob,1,&cv); mrw_baked_view_at(&blob,2,&bv);
    uint32_t fst = jc*2;

    /* (1) t = duration wraps to baked frame 0 ⇒ identical to t = 0 */
    for (uint32_t b=0;b<jc;++b){
        float a0[12], ad[12];
        mrw_baked_sample_bone(&bv,0,b,0.0f,a0);
        mrw_baked_sample_bone(&bv,0,b,dur, ad);
        CHECK(aff_near(a0, ad, 1e-6));
    }
    /* (2) negative t wraps; t=-eps ≈ t=dur-eps */
    float eps = dur*1e-3f;
    for (uint32_t b=0;b<jc;++b){
        float an[12], ad[12];
        mrw_baked_sample_bone(&bv,0,b,-eps,an);
        mrw_baked_sample_bone(&bv,0,b,dur-eps,ad);
        CHECK(aff_near(an, ad, 1e-4));
    }
    /* (3) runtime baked path == independent component-space reference at assorted phases,
     *     including the endpoints - proves sampling/blend agreement both ways. */
    float worst=0.0f; float phases[] = {0.0f, eps, dur*0.37f, dur*0.5f, dur-eps, dur, dur*1.5f, -eps};
    for (uint32_t i=0;i<sizeof phases/sizeof phases[0];++i){
        float t = phases[i];
        for (uint32_t b=0;b<jc;++b){
            float affR[12]; mrw_baked_sample_bone(&bv,0,b,t,affR);
            mrw_xform xo; mrw_bake_sample_xform(g_tex, fst, 0, fc, dur, 1, b, t, &xo);
            float affO[12]; mrw_xform_to_affine(&xo, affO);
            for (int k=0;k<12;k++){ float d=fabsf(affR[k]-affO[k]); if(d>worst)worst=d; }
            CHECK(aff_near(affR, affO, 1e-5));
        }
    }
    printf("  [E] looping endpoints: wrap continuous; runtime-vs-oracle worst |Δaffine| = %.2e\n", worst);
    mrw_free(buf);
}

/* ===================================================================== F: promotion-pop + sign */
static void test_promotion_and_sign(void) {
    uint32_t jc=10, sc=13;
    gen_rig(jc, sc, 0x44, -1, 0.4f, g_parent, g_rest, g_ib, g_samp);
    make_probes(jc, g_parent, g_rest, g_probes);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { FPS, sc, 0, g_samp, NULL };
    float dur = (float)(sc-1)/FPS;
    uint32_t fc = sc;     /* one baked frame per keyframe ⇒ small between-frame pop */

    mrw_blob tmp; uint8_t *btmp = open_blob(&skel, &clip, 1, NULL, &tmp);
    float resid; CHECK_EQ(bake_into(&tmp, 1, fc, 0, g_tex, 1e-3f, &resid), 1);
    mrw_free(btmp);
    uint32_t bidx[1]={0}, bff[1]={0}, bfc[1]={fc}, bflags[1]={0}; float bdur[1]={dur};
    mrw_baked baked = { 0, fc, g_tex, 1, bidx, bff, bfc, bdur, bflags };
    mrw_blob blob; uint8_t *buf = open_blob(&skel, &clip, 1, &baked, &blob);
    mrw_skeleton_view sv; mrw_clip_view cv; mrw_baked_view bv;
    mrw_blob_skeleton(&blob,&sv); mrw_clip_view_at(&blob,1,&cv); mrw_baked_view_at(&blob,2,&bv);

    /* promotion pop at arbitrary (off-grid) phases */
    rng r = { 0x9e3779b9u };
    float worst=0.0f;
    for (int i=0;i<200;i++){
        float t = rf(&r, 0.0f, dur);
        float p = tierAB_pop(&sv,&cv,&bv,0,t,g_probes);
        if (p>worst) worst=p;
    }
    CHECK(worst < 3e-2);   /* bounded, visually small at clip-rate bake (measured ~1.4cm) */
    printf("  [F] promotion pop @ arbitrary phase: worst=%.3e m over 200 phases\n", worst);

    /* direct baked sign-continuity: adjacent frames share a hemisphere AFTER decode
     * (the runtime nlerp also corrects, so this must be checked on the raw texels). */
    uint32_t fst = jc*2; float min_dot=1.0f;
    for (uint32_t b=0;b<jc;++b)
        for (uint32_t f=1; f<fc; ++f){
            mrw_xform a,bq; mrw_bake_decode(g_tex,fst,f-1,b,&a); mrw_bake_decode(g_tex,fst,f,b,&bq);
            float dot=a.rot[0]*bq.rot[0]+a.rot[1]*bq.rot[1]+a.rot[2]*bq.rot[2]+a.rot[3]*bq.rot[3];
            if (dot<min_dot) min_dot=dot;
            CHECK(dot >= -1e-6f);
        }
    printf("  [F] baked sign-continuity: min adjacent dot = %.4f (>= 0)\n", min_dot);
    mrw_free(buf);
}

/* ===================================================================== G: cross-fade gap */
static void test_crossfade(void) {
    uint32_t jc=6, sc=9;
    gen_rig(jc, sc, 0x55, -1, 0.4f, g_parent, g_rest, g_ib, g_samp);   /* clip A */
    fill_smooth_clip(jc, sc, 0x66u, 0.4f, g_samp2);                    /* clip B (same skeleton) */
    make_probes(jc, g_parent, g_rest, g_probes);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clips[2] = { { FPS, sc, 0, g_samp, NULL }, { FPS, sc, 0, g_samp2, NULL } };
    float dur = (float)(sc-1)/FPS;
    uint32_t fc = sc, fst = jc*2;

    mrw_blob tmp; uint8_t *btmp = open_blob(&skel, clips, 2, NULL, &tmp);
    float resid;
    bake_into(&tmp, 1, fc, 0,  g_tex, 1e-3f, &resid);   /* clip A → frames [0,fc)   */
    bake_into(&tmp, 2, fc, fc, g_tex, 1e-3f, &resid);   /* clip B → frames [fc,2fc) */
    mrw_skeleton_view sv; mrw_clip_view cv0, cv1;
    mrw_blob_skeleton(&tmp,&sv); mrw_clip_view_at(&tmp,1,&cv0); mrw_clip_view_at(&tmp,2,&cv1);

    rng r = { 0x1234u };
    float worst=0.0f;
    for (int i=0;i<120;i++){
        float t = rf(&r, 0.0f, dur);
        float w = rf(&r, 0.0f, 1.0f);
        /* CPU-tier reference: blend LOCALS (nlerp+lerp), then local→model→palette */
        mrw_trs la[MAXJ], lb[MAXJ], lblend[MAXJ];
        mrw_clip_sample_local(&cv0, t, la, jc);
        mrw_clip_sample_local(&cv1, t, lb, jc);
        for (uint32_t j=0;j<jc;++j){
            mrw_quat_nlerp(la[j].rot, lb[j].rot, w, lblend[j].rot);
            for (int k=0;k<3;k++){ lblend[j].trans[k]=la[j].trans[k]+w*(lb[j].trans[k]-la[j].trans[k]);
                                   lblend[j].scale[k]=la[j].scale[k]+w*(lb[j].scale[k]-la[j].scale[k]); }
        }
        float modelX[MAXJ*12], palX[MAXJ*12];
        mrw_local_to_model(&sv, lblend, modelX, jc);
        mrw_model_to_palette(&sv, modelX, palX, jc);
        /* Baked tier: decode both clips in Q/T/s space, temporal-sample each, cross-clip
         * nlerp at weight w, THEN compose (interpolate before composing the 3×4). */
        for (uint32_t b=0;b<jc;++b){
            mrw_xform xa, xb, xc;
            mrw_bake_sample_xform(g_tex, fst, 0,  fc, dur, 0, b, t, &xa);
            mrw_bake_sample_xform(g_tex, fst, fc, fc, dur, 0, b, t, &xb);
            mrw_xform_nlerp(&xa, &xb, w, &xc);
            float affB[12]; mrw_xform_to_affine(&xc, affB);
            float d = mrw_affine_probe_dist(palX+(size_t)b*12, affB, NP, g_probes+(size_t)b*NP*3);
            if (d>worst) worst=d;
        }
    }
    CHECK(worst < 0.5f);   /* bounded: the named, sanctioned cross-fade approximation */
    printf("  [G] cross-fade gap (component-space blend vs local-blend): worst=%.3e m\n", worst);
    mrw_free(btmp);
}

/* ===================================================================== H: whole-rig eligibility */
static void test_eligibility(void) {
    uint32_t jc=5, sc=4;
    /* rigid baseline ⇒ eligible */
    gen_rig(jc, sc, 0x77, -1, 0.5f, g_parent, g_rest, g_ib, g_samp);
    make_probes(jc, g_parent, g_rest, g_probes);
    mrw_skel sk0 = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip cl0 = { FPS, sc, 0, g_samp, NULL };
    mrw_blob t0; uint8_t *b0 = open_blob(&sk0, &cl0, 1, NULL, &t0);
    float r0; int e0 = bake_into(&t0, 1, sc, 0, g_tex, 1e-3f, &r0);
    mrw_free(b0);
    CHECK_EQ(e0, 1);

    /* one non-uniform-scale bone ⇒ its skinning transform is not a similarity ⇒ WHOLE rig rejected */
    gen_rig(jc, sc, 0x77, /*nonuniform_bone*/2, 0.5f, g_parent, g_rest, g_ib, g_samp);
    make_probes(jc, g_parent, g_rest, g_probes);
    mrw_skel sk1 = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip cl1 = { FPS, sc, 0, g_samp, NULL };
    mrw_blob t1; uint8_t *b1 = open_blob(&sk1, &cl1, 1, NULL, &t1);
    float r1; int e1 = bake_into(&t1, 1, sc, 0, g_tex, 1e-3f, &r1);
    mrw_free(b1);
    CHECK_EQ(e1, 0);
    CHECK(r1 > 1e-3f);    /* the worst residual exceeded tol */

    /* a bone with NO probes is eligible by default - same non-uniform rig, np=0
     * (probes NULL) ⇒ the rig is accepted because nothing measurable deforms. */
    mrw_blob t2; uint8_t *b2 = open_blob(&sk1, &cl1, 1, NULL, &t2);
    mrw_skeleton_view sv2; mrw_clip_view cv2;
    mrw_blob_skeleton(&t2,&sv2); mrw_clip_view_at(&t2,1,&cv2);
    mrw_mem_req sreq2, treq2;
    CHECK_EQ(mrw_bake_clip_requirements(sv2.joint_count, sc, &sreq2, &treq2), MRW_OK);
    mrw_bake_stats st2;
    CHECK_EQ(mrw_bake_clip(&sv2, &cv2, sc, NULL, NULL, 1e-3f,
                           g_bake_scratch, sizeof g_bake_scratch, g_tex, treq2.size, &st2), MRW_OK);
    int e2 = st2.eligible; float r2 = st2.max_residual;
    mrw_free(b2);
    CHECK_EQ(e2, 1);
    CHECK_NEAR(r2, 0.0f, 0.0);
    printf("  [H] eligibility: rigid eligible; non-uniform rejects whole rig (residual=%.3e m); no-probe eligible-by-default\n", r1);
}

int main(void) {
    test_mat3_to_quat();
    test_decompose_checks();
    test_exact_frame();
    test_between_frame();
    test_looping();
    test_promotion_and_sign();
    test_crossfade();
    test_eligibility();
    printf(g_fail ? "test_parity: %d FAILED\n" : "test_parity: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
