/* marrow-bake end-to-end + bake-core test. Rigs are built in-memory through the checked authoring
 * writer (no committed binaries); the front-end is exercised over a real temp .mrw file (mrw_bake_run
 * reads a path), and the bake core is exercised directly for the caller-owned-buffer contract and the
 * all-texel-finite rejection that the box-probe front-end can't reach (every bone is probed there).
 *
 * Coverage: eligible exact-frame parity + well-formed BAKED header; ineligible non-uniform rig stays
 * a clip-set-only copy; frame-count/duration invariants (static ⇒ 1 frame/dur 0; short-dynamic ⇒ ≥2
 * frames/dur>0; looping flag carried); bad --bake-fps; --decompose-tolerance flip; determinism; buffer
 * contract; unprobed-bone quantization reject. */
#include "bake_run.h"
#include "mrw_bake.h"
#include "mrw_authoring.h"
#include "mrw_decompose.h"   /* mrw_affine_probe_dist */
#include "marrow.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MRW_BAKE_TESTDIR
#define MRW_BAKE_TESTDIR "."
#endif

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)
#define CHECK_NEAR(a,b,eps) CHECK(fabs((double)(a)-(double)(b)) <= (eps))

#define MAXJ 16u
#define MAXS 64u
#define NP   8u

static const char *NAMES[MAXJ] = {
    "j0","j1","j2","j3","j4","j5","j6","j7","j8","j9","j10","j11","j12","j13","j14","j15"
};

/* invert a 3x4 affine [A|t] -> [A^-1 | -A^-1 t]; returns 0 if singular (cofactor / det). */
static int aff_inverse(const float m[12], float out[12]) {
    float a[9] = { m[0],m[1],m[2], m[4],m[5],m[6], m[8],m[9],m[10] };
    double det = (double)a[0]*(a[4]*a[8]-a[5]*a[7]) - (double)a[1]*(a[3]*a[8]-a[5]*a[6]) + (double)a[2]*(a[3]*a[7]-a[4]*a[6]);
    if (fabs(det) < 1e-20) return 0;
    float c[9]; mrw_cofactor3(a, c);
    float ai[9];
    for (int r = 0; r < 3; ++r) for (int col = 0; col < 3; ++col) ai[3*r+col] = (float)(c[3*col+r] / det);
    float t[3] = { m[3], m[7], m[11] };
    out[0]=ai[0]; out[1]=ai[1]; out[2]=ai[2];  out[3]=-(ai[0]*t[0]+ai[1]*t[1]+ai[2]*t[2]);
    out[4]=ai[3]; out[5]=ai[4]; out[6]=ai[5];  out[7]=-(ai[3]*t[0]+ai[4]*t[1]+ai[5]*t[2]);
    out[8]=ai[6]; out[9]=ai[7]; out[10]=ai[8]; out[11]=-(ai[6]*t[0]+ai[7]*t[1]+ai[8]*t[2]);
    return 1;
}

static void rest_to_model(uint32_t jc, const uint16_t *parent, const float *rest, float *model) {
    for (uint32_t j = 0; j < jc; ++j) {
        mrw_trs trs;
        memcpy(trs.rot, rest+j*10+0, 16); memcpy(trs.trans, rest+j*10+4, 12); memcpy(trs.scale, rest+j*10+7, 12);
        float local[12]; mrw_trs_to_affine(&trs, local);
        if (parent[j] == 0xFFFFu) memcpy(model+(size_t)j*12, local, sizeof local);
        else mrw_affine_mul(model+(size_t)parent[j]*12, local, model+(size_t)j*12);
    }
}

static void compute_bind_inverse(uint32_t jc, const uint16_t *parent, const float *rest, float *ib) {
    static float model[MAXJ*12];
    rest_to_model(jc, parent, rest, model);
    for (uint32_t j = 0; j < jc; ++j) aff_inverse(model+(size_t)j*12, ib+(size_t)j*12);
}

/* 8 box corners (±r) at each bone's bind-pose model position - independent parity-measurement probes. */
static void make_probes(uint32_t jc, const uint16_t *parent, const float *rest, float r, float *out) {
    static float model[MAXJ*12];
    rest_to_model(jc, parent, rest, model);
    for (uint32_t j = 0; j < jc; ++j) {
        float bx=model[j*12+3], by=model[j*12+7], bz=model[j*12+11];
        uint32_t k=0;
        for (int sx=-1;sx<=1;sx+=2) for (int sy=-1;sy<=1;sy+=2) for (int sz=-1;sz<=1;sz+=2) {
            float *p = out + ((size_t)j*NP + k)*3; p[0]=bx+sx*r; p[1]=by+sy*r; p[2]=bz+sz*r; ++k;
        }
    }
}

/* Build a rigid chain rig (parent[j]=j-1) + a smooth rotation clip. `nonuniform_bone` (or -1) gives
 * that bone a non-uniform REST scale (its skinning transform stops being a similarity). `bigtrans_bone`
 * (or -1) gives that bone a huge clip translation (its palette translation overflows binary16). */
static void build_chain(uint32_t jc, uint32_t sc, int nonuniform_bone, int bigtrans_bone, float amp,
                        uint16_t *parent, float *rest, float *ib, float *samp) {
    for (uint32_t j = 0; j < jc; ++j) {
        parent[j] = (j==0) ? 0xFFFFu : (uint16_t)(j-1);
        rest[j*10+0]=0; rest[j*10+1]=0; rest[j*10+2]=0; rest[j*10+3]=1;   /* identity rest rotation */
        rest[j*10+4]=(j==0)?0.0f:0.22f; rest[j*10+5]=0.04f; rest[j*10+6]=-0.03f;
        rest[j*10+7]=rest[j*10+8]=rest[j*10+9]=1.0f;
        if ((int)j == nonuniform_bone) { rest[j*10+7]=2.0f; rest[j*10+8]=0.5f; rest[j*10+9]=1.3f; }
    }
    compute_bind_inverse(jc, parent, rest, ib);
    for (uint32_t j = 0; j < jc; ++j) {
        float ax=0.2f+0.1f*j, ay=1.0f, az=0.3f*j; float n=sqrtf(ax*ax+ay*ay+az*az); ax/=n; ay/=n; az/=n;
        float freq=0.5f+0.05f*j, ph=0.3f*j;
        for (uint32_t s = 0; s < sc; ++s) {
            float u = (sc>1) ? (float)s/(float)(sc-1) : 0.0f;
            float ang = amp * sinf(6.2831853f*freq*u + ph);
            float sh = sinf(ang*0.5f), ch = cosf(ang*0.5f);
            float *smp = samp + ((size_t)j*sc + s)*10;
            smp[0]=ax*sh; smp[1]=ay*sh; smp[2]=az*sh; smp[3]=ch;
            smp[4]=0.02f*sinf(u*3.0f+j); smp[5]=0.0f; smp[6]=0.0f;
            smp[7]=smp[8]=smp[9]=1.0f;
            if ((int)j == bigtrans_bone) { smp[4]=1.0e5f; smp[5]=0.0f; smp[6]=0.0f; }
        }
    }
}

/* Serialize a {skel, clips} clip set to `path` (no baked section). Returns 0 on success. */
static int write_clipset(const char *path, const mrw_skel *skel, const mrw_clip *clips, uint32_t nclip) {
    uint8_t *buf=NULL; size_t size=0;
    if (mrw_authoring_build(skel, clips, nclip, NULL, &buf, &size) != MRW_OK) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { mrw_authoring_free(buf); return -1; }
    size_t w = fwrite(buf, 1, size, f);
    int err = (w != size); if (fclose(f) != 0) err = 1;
    mrw_authoring_free(buf);
    return err ? -1 : 0;
}

/* ---- scratch ---- */
static uint16_t g_parent[MAXJ];
static float    g_rest[MAXJ*10], g_ib[MAXJ*12], g_samp[MAXJ*MAXS*10];
static float    g_probes[MAXJ*NP*3];
static float    g_model[MAXJ*12], g_palA[MAXJ*12];

/* ===================================================================== eligible exact-frame parity */
static int test_eligible_parity(void) {
    uint32_t jc=6, sc=6;
    build_chain(jc, sc, -1, -1, 0.7f, g_parent, g_rest, g_ib, g_samp);
    make_probes(jc, g_parent, g_rest, 0.15f, g_probes);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { 30.0f, sc, 0, g_samp, NULL };
    const char *path = MRW_BAKE_TESTDIR "/bake_elig.mrw";
    CHECK(write_clipset(path, &skel, &clip, 1) == 0);

    mrw_bake_options opt = {0}; opt.input_path = path; opt.bake_fps = 30.0f; opt.mesh_skin_index = -1;
    uint8_t *obuf=NULL; size_t osize=0; int eligible=0; mrw_bake_stats worst; char diag[256]={0};
    CHECK(mrw_bake_run(&opt, &obuf, &osize, &eligible, &worst, diag, sizeof diag) == MRW_OK);
    CHECK(eligible == 1);

    mrw_blob blob; CHECK(mrw_blob_open(obuf, (uint64_t)osize, &blob) == MRW_OK);
    mrw_skeleton_view sv; mrw_clip_view cv; mrw_baked_view bv;
    CHECK(mrw_blob_skeleton(&blob, &sv) == MRW_OK);
    /* locate the clip + baked sections by type */
    int have_clip=0, have_baked=0;
    for (uint32_t i=0;i<blob.section_count;++i){ uint32_t ty=0; mrw_blob_section_type(&blob,i,&ty);
        if (ty==MRW_SECTION_CLIP && !have_clip){ CHECK(mrw_clip_view_at(&blob,i,&cv)==MRW_OK); have_clip=1; }
        if (ty==MRW_SECTION_BAKED){ CHECK(mrw_baked_view_at(&blob,i,&bv)==MRW_OK); have_baked=1; } }
    CHECK(have_clip && have_baked);

    /* well-formed BAKED header */
    CHECK(bv.encoding == 1); CHECK(bv.texels_per_bone == 2); CHECK(bv.bone_count == jc);
    CHECK(bv.frame_stride_texels == jc*2);
    CHECK((uint64_t)bv.total_frames * bv.frame_stride_texels == (uint64_t)bv.texel_count);
    CHECK(bv.clip_count == 1);
    mrw_baked_clip ce; CHECK(mrw_baked_clip_entry(&bv, 0, &ce) == MRW_OK);
    CHECK(ce.frame_count == sc);              /* bake_fps == clip fps ⇒ one frame per keyframe */
    CHECK(ce.first_frame == 0);
    /* clip_id resolves to the paired CLIP */
    CHECK(mrw_id_equal(&ce.clip_id, &cv.id));

    /* exact-frame parity: baked sample vs CPU sample at every baked frame, sub-2mm (binary16) */
    float dur = ce.source_duration_s; uint32_t fc = ce.frame_count;
    float maxe = 0.0f;
    for (uint32_t f=0; f<fc; ++f) {
        float t = (fc<2)?0.0f:(float)f/(float)(fc-1)*dur;
        CHECK(mrw_clip_to_palette(&sv,&cv,t,g_model,g_palA,jc) == MRW_OK);
        for (uint32_t b=0;b<jc;++b){
            float affB[12]; CHECK(mrw_baked_sample_bone(&bv,0,b,t,affB)==MRW_OK);
            float e = mrw_affine_probe_dist(g_palA+(size_t)b*12, affB, NP, g_probes+(size_t)b*NP*3);
            if (e>maxe) maxe=e;
        }
    }
    CHECK(maxe < 2.0e-3f);
    printf("  [bake] eligible parity: worst %.3e m, %u frames; %s\n", maxe, fc, diag);
    mrw_authoring_free(obuf);
    return 0;
}

/* ===================================================================== ineligible non-uniform rig */
static int test_ineligible(void) {
    uint32_t jc=5, sc=5;
    build_chain(jc, sc, /*nonuniform*/2, -1, 0.6f, g_parent, g_rest, g_ib, g_samp);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { 30.0f, sc, 0, g_samp, NULL };
    const char *path = MRW_BAKE_TESTDIR "/bake_inelig.mrw";
    CHECK(write_clipset(path, &skel, &clip, 1) == 0);

    mrw_bake_options opt = {0}; opt.input_path = path; opt.bake_fps = 30.0f; opt.mesh_skin_index = -1;
    opt.decompose_tol = 1.0e-3f;
    uint8_t *obuf=NULL; size_t osize=0; int eligible=1; mrw_bake_stats worst; char diag[256]={0};
    CHECK(mrw_bake_run(&opt, &obuf, &osize, &eligible, &worst, diag, sizeof diag) == MRW_OK);
    CHECK(eligible == 0);
    CHECK(worst.reason == MRW_BAKE_RESIDUAL);
    CHECK(worst.max_residual > 1.0e-3f);

    /* output is a valid .mrw with NO BAKED section */
    mrw_blob blob; CHECK(mrw_blob_open(obuf, (uint64_t)osize, &blob) == MRW_OK);
    int n_baked=0, n_clip=0;
    for (uint32_t i=0;i<blob.section_count;++i){ uint32_t ty=0; mrw_blob_section_type(&blob,i,&ty);
        if (ty==MRW_SECTION_BAKED) ++n_baked; if (ty==MRW_SECTION_CLIP) ++n_clip; }
    CHECK(n_baked == 0); CHECK(n_clip == 1);
    printf("  [bake] ineligible: %s\n", diag);
    mrw_authoring_free(obuf);
    return 0;
}

/* ===================================================================== frame-count / duration */
static int test_frame_counts(void) {
    /* (a) static clip (1 sample) ⇒ 1 baked frame, source_duration 0 */
    {
        uint32_t jc=3, sc=1;
        build_chain(jc, sc, -1, -1, 0.0f, g_parent, g_rest, g_ib, g_samp);
        mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
        mrw_clip clip = { 30.0f, sc, 0, g_samp, NULL };
        const char *path = MRW_BAKE_TESTDIR "/bake_static.mrw";
        CHECK(write_clipset(path, &skel, &clip, 1) == 0);
        mrw_bake_options opt = {0}; opt.input_path=path; opt.bake_fps=30.0f; opt.mesh_skin_index=-1;
        uint8_t *obuf=NULL; size_t osize=0; int el=0; mrw_bake_stats w; char d[256]={0};
        CHECK(mrw_bake_run(&opt,&obuf,&osize,&el,&w,d,sizeof d)==MRW_OK);
        CHECK(el==1);
        mrw_blob b; CHECK(mrw_blob_open(obuf,(uint64_t)osize,&b)==MRW_OK);
        mrw_baked_view bv; int found=0;
        for (uint32_t i=0;i<b.section_count;++i){ uint32_t ty=0; mrw_blob_section_type(&b,i,&ty); if(ty==MRW_SECTION_BAKED){mrw_baked_view_at(&b,i,&bv);found=1;} }
        CHECK(found);
        mrw_baked_clip ce; CHECK(mrw_baked_clip_entry(&bv,0,&ce)==MRW_OK);
        CHECK(ce.frame_count == 1); CHECK(ce.source_duration_s == 0.0f);
        mrw_authoring_free(obuf);
    }
    /* (b) short dynamic at low bake-fps ⇒ still ≥2 frames, source_duration > 0 */
    {
        uint32_t jc=3, sc=2;
        build_chain(jc, sc, -1, -1, 0.4f, g_parent, g_rest, g_ib, g_samp);
        mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
        mrw_clip clip = { 30.0f, sc, 0, g_samp, NULL };          /* dur = 1/30 s */
        const char *path = MRW_BAKE_TESTDIR "/bake_short.mrw";
        CHECK(write_clipset(path, &skel, &clip, 1) == 0);
        mrw_bake_options opt = {0}; opt.input_path=path; opt.bake_fps=1.0f; opt.mesh_skin_index=-1; /* round(0.033)+1 = 1 → max(2,..) */
        uint8_t *obuf=NULL; size_t osize=0; int el=0; mrw_bake_stats w; char d[256]={0};
        CHECK(mrw_bake_run(&opt,&obuf,&osize,&el,&w,d,sizeof d)==MRW_OK);
        CHECK(el==1);
        mrw_blob b; CHECK(mrw_blob_open(obuf,(uint64_t)osize,&b)==MRW_OK);
        mrw_baked_view bv; int found=0;
        for (uint32_t i=0;i<b.section_count;++i){ uint32_t ty=0; mrw_blob_section_type(&b,i,&ty); if(ty==MRW_SECTION_BAKED){mrw_baked_view_at(&b,i,&bv);found=1;} }
        CHECK(found);
        mrw_baked_clip ce; CHECK(mrw_baked_clip_entry(&bv,0,&ce)==MRW_OK);
        CHECK(ce.frame_count >= 2); CHECK(ce.source_duration_s > 0.0f);
        mrw_authoring_free(obuf);
    }
    /* (c) looping clip ⇒ baked clip carries MRW_BAKED_CLIP_LOOPING */
    {
        uint32_t jc=4, sc=8;
        build_chain(jc, sc, -1, -1, 0.5f, g_parent, g_rest, g_ib, g_samp);
        mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
        mrw_clip clip = { 30.0f, sc, MRW_CLIP_LOOPING, g_samp, NULL };
        const char *path = MRW_BAKE_TESTDIR "/bake_loop.mrw";
        CHECK(write_clipset(path, &skel, &clip, 1) == 0);
        mrw_bake_options opt = {0}; opt.input_path=path; opt.bake_fps=30.0f; opt.mesh_skin_index=-1;
        uint8_t *obuf=NULL; size_t osize=0; int el=0; mrw_bake_stats w; char d[256]={0};
        CHECK(mrw_bake_run(&opt,&obuf,&osize,&el,&w,d,sizeof d)==MRW_OK);
        CHECK(el==1);
        mrw_blob b; CHECK(mrw_blob_open(obuf,(uint64_t)osize,&b)==MRW_OK);
        mrw_baked_view bv; int found=0;
        for (uint32_t i=0;i<b.section_count;++i){ uint32_t ty=0; mrw_blob_section_type(&b,i,&ty); if(ty==MRW_SECTION_BAKED){mrw_baked_view_at(&b,i,&bv);found=1;} }
        CHECK(found);
        mrw_baked_clip ce; CHECK(mrw_baked_clip_entry(&bv,0,&ce)==MRW_OK);
        CHECK((ce.flags & MRW_BAKED_CLIP_LOOPING) != 0);
        mrw_authoring_free(obuf);
    }
    printf("  [bake] frame-count/duration invariants: static/short-dynamic/looping ok\n");
    return 0;
}

/* ===================================================================== bad --bake-fps */
static int test_bad_fps(void) {
    uint32_t jc=3, sc=4;
    build_chain(jc, sc, -1, -1, 0.4f, g_parent, g_rest, g_ib, g_samp);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { 30.0f, sc, 0, g_samp, NULL };
    const char *path = MRW_BAKE_TESTDIR "/bake_badfps.mrw";
    CHECK(write_clipset(path, &skel, &clip, 1) == 0);

    mrw_bake_options opt = {0}; opt.input_path=path; opt.mesh_skin_index=-1;
    uint8_t *obuf=(uint8_t*)1; size_t osize=0; int el=0; mrw_bake_stats w; char d[256]={0};
    opt.bake_fps = 1e30f;                                  /* yields too many frames */
    CHECK(mrw_bake_run(&opt,&obuf,&osize,&el,&w,d,sizeof d) != MRW_OK);
    CHECK(obuf == NULL);                                   /* nothing allocated on error */
    opt.bake_fps = 0.0f;                                   /* non-positive */
    CHECK(mrw_bake_run(&opt,&obuf,&osize,&el,&w,d,sizeof d) == MRW_E_RANGE);
    printf("  [bake] bad --bake-fps rejected cleanly\n");
    return 0;
}

/* ===================================================================== --decompose-tolerance flip */
static int test_tolerance_flip(void) {
    uint32_t jc=4, sc=5;
    build_chain(jc, sc, /*nonuniform*/1, -1, 0.6f, g_parent, g_rest, g_ib, g_samp);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { 30.0f, sc, 0, g_samp, NULL };
    const char *path = MRW_BAKE_TESTDIR "/bake_tol.mrw";
    CHECK(write_clipset(path, &skel, &clip, 1) == 0);

    uint8_t *obuf=NULL; size_t osize=0; int el=0; mrw_bake_stats w; char d[256]={0};
    mrw_bake_options opt = {0}; opt.input_path=path; opt.bake_fps=30.0f; opt.mesh_skin_index=-1; opt.probe_radius=0.1f;

    opt.decompose_tol = 1.0e-4f;     /* tight ⇒ the non-uniform bone fails */
    CHECK(mrw_bake_run(&opt,&obuf,&osize,&el,&w,d,sizeof d) == MRW_OK);
    CHECK(el == 0); mrw_authoring_free(obuf); obuf=NULL;

    opt.decompose_tol = 1.0e3f;      /* loose ⇒ accepted */
    CHECK(mrw_bake_run(&opt,&obuf,&osize,&el,&w,d,sizeof d) == MRW_OK);
    CHECK(el == 1); mrw_authoring_free(obuf); obuf=NULL;
    printf("  [bake] --decompose-tolerance flips eligibility\n");
    return 0;
}

/* ===================================================================== determinism */
static int test_determinism(void) {
    uint32_t jc=5, sc=7;
    build_chain(jc, sc, -1, -1, 0.5f, g_parent, g_rest, g_ib, g_samp);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { 30.0f, sc, 0, g_samp, NULL };
    const char *path = MRW_BAKE_TESTDIR "/bake_det.mrw";
    CHECK(write_clipset(path, &skel, &clip, 1) == 0);

    mrw_bake_options opt = {0}; opt.input_path=path; opt.bake_fps=24.0f; opt.mesh_skin_index=-1;
    uint8_t *a=NULL,*b=NULL; size_t sa=0,sb=0; int ea=0,eb=0; mrw_bake_stats wa,wb; char d[256]={0};
    CHECK(mrw_bake_run(&opt,&a,&sa,&ea,&wa,d,sizeof d)==MRW_OK);
    CHECK(mrw_bake_run(&opt,&b,&sb,&eb,&wb,d,sizeof d)==MRW_OK);
    CHECK(ea==1 && eb==1);
    CHECK(sa==sb);
    CHECK(memcmp(a,b,sa)==0);
    mrw_authoring_free(a); mrw_authoring_free(b);
    printf("  [bake] determinism: byte-identical re-bake (%zu bytes)\n", sa);
    return 0;
}

/* ===================================================================== bake-core buffer contract */
static int test_buffer_contract(void) {
    uint32_t jc=4, sc=4, fc=4;
    build_chain(jc, sc, -1, -1, 0.4f, g_parent, g_rest, g_ib, g_samp);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { 30.0f, sc, 0, g_samp, NULL };
    uint8_t *buf=NULL; size_t size=0;
    CHECK(mrw_authoring_build(&skel,&clip,1,NULL,&buf,&size)==MRW_OK);
    mrw_blob blob; CHECK(mrw_blob_open(buf,(uint64_t)size,&blob)==MRW_OK);
    mrw_skeleton_view sv; mrw_clip_view cv;
    CHECK(mrw_blob_skeleton(&blob,&sv)==MRW_OK);
    for (uint32_t i=0;i<blob.section_count;++i){ uint32_t ty=0; mrw_blob_section_type(&blob,i,&ty); if(ty==MRW_SECTION_CLIP){mrw_clip_view_at(&blob,i,&cv);break;} }

    /* requirements: frame_count==0 ⇒ MRW_E_RANGE; normal ⇒ OK */
    mrw_mem_req sreq, treq;
    CHECK(mrw_bake_clip_requirements(jc, 0, &sreq, &treq) == MRW_E_RANGE);
    CHECK(mrw_bake_clip_requirements(jc, fc, &sreq, &treq) == MRW_OK);
    CHECK(sreq.align == 16 && treq.align == 16);

    void *scratch = mrw_authoring_alloc(sreq.size + 16);
    uint16_t *tex = (uint16_t *)mrw_authoring_alloc(treq.size + 16);
    CHECK(scratch && tex);
    for (uint32_t j=0;j<jc;++j) g_parent[j]=g_parent[j]; /* no-op to keep g_parent referenced */
    uint32_t pc[MAXJ]; for (uint32_t j=0;j<jc;++j) pc[j]=0;   /* no probes (we only test buffers) */
    mrw_bake_stats st;

    /* frame_count==0 ⇒ RANGE */
    CHECK(mrw_bake_clip(&sv,&cv,0,pc,NULL,1e-3f,scratch,sreq.size,tex,treq.size,&st) == MRW_E_RANGE);
    /* undersized texels / scratch ⇒ CAPACITY */
    CHECK(mrw_bake_clip(&sv,&cv,fc,pc,NULL,1e-3f,scratch,sreq.size,tex,treq.size-1,&st) == MRW_E_CAPACITY);
    CHECK(mrw_bake_clip(&sv,&cv,fc,pc,NULL,1e-3f,scratch,sreq.size-1,tex,treq.size,&st) == MRW_E_CAPACITY);
    /* misaligned texels / scratch (capacity ok) ⇒ ALIGN */
    CHECK(mrw_bake_clip(&sv,&cv,fc,pc,NULL,1e-3f,scratch,sreq.size,(uint16_t*)((uint8_t*)tex+8),treq.size,&st) == MRW_E_ALIGN);
    CHECK(mrw_bake_clip(&sv,&cv,fc,pc,NULL,1e-3f,(uint8_t*)scratch+8,sreq.size,tex,treq.size,&st) == MRW_E_ALIGN);
    /* well-formed call ⇒ OK */
    CHECK(mrw_bake_clip(&sv,&cv,fc,pc,NULL,1e-3f,scratch,sreq.size,tex,treq.size,&st) == MRW_OK);
    CHECK(st.eligible == 1);   /* no probes ⇒ eligible by default, finite texels */

    mrw_authoring_free(scratch); mrw_authoring_free(tex); mrw_authoring_free(buf);
    printf("  [bake] buffer contract: RANGE/CAPACITY/ALIGN enforced\n");
    return 0;
}

/* ===================================================================== unprobed-bone quantization */
static int test_quantization_reject(void) {
    uint32_t jc=4, sc=4, fc=4;
    build_chain(jc, sc, -1, /*bigtrans*/2, 0.4f, g_parent, g_rest, g_ib, g_samp);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { 30.0f, sc, 0, g_samp, NULL };
    uint8_t *buf=NULL; size_t size=0;
    CHECK(mrw_authoring_build(&skel,&clip,1,NULL,&buf,&size)==MRW_OK);
    mrw_blob blob; CHECK(mrw_blob_open(buf,(uint64_t)size,&blob)==MRW_OK);
    mrw_skeleton_view sv; mrw_clip_view cv;
    CHECK(mrw_blob_skeleton(&blob,&sv)==MRW_OK);
    for (uint32_t i=0;i<blob.section_count;++i){ uint32_t ty=0; mrw_blob_section_type(&blob,i,&ty); if(ty==MRW_SECTION_CLIP){mrw_clip_view_at(&blob,i,&cv);break;} }

    mrw_mem_req sreq, treq; CHECK(mrw_bake_clip_requirements(jc, fc, &sreq, &treq)==MRW_OK);
    void *scratch = mrw_authoring_alloc(sreq.size);
    uint16_t *tex = (uint16_t *)mrw_authoring_alloc(treq.size);
    CHECK(scratch && tex);
    uint32_t pc[MAXJ]; for (uint32_t j=0;j<jc;++j) pc[j]=0;   /* the overflow bone (2) is UNPROBED */
    mrw_bake_stats st;
    CHECK(mrw_bake_clip(&sv,&cv,fc,pc,NULL,1e-3f,scratch,sreq.size,tex,treq.size,&st) == MRW_OK);
    CHECK(st.eligible == 0);                       /* non-finite texel rejects even an unprobed bone */
    CHECK(st.reason == MRW_BAKE_QUANTIZED);

    mrw_authoring_free(scratch); mrw_authoring_free(tex); mrw_authoring_free(buf);
    printf("  [bake] quantization: unprobed binary16-overflow bone rejects rig\n");
    return 0;
}

/* ===================================================================== --mesh probe extraction */
static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static void base64(const uint8_t *in, size_t n, char *out) {
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t b0=in[i], b1=(i+1<n)?in[i+1]:0, b2=(i+2<n)?in[i+2]:0;
        uint32_t t=(b0<<16)|(b1<<8)|b2;
        out[o++]=B64[(t>>18)&63]; out[o++]=B64[(t>>12)&63];
        out[o++]=(i+1<n)?B64[(t>>6)&63]:'='; out[o++]=(i+2<n)?B64[t&63]:'=';
    }
    out[o]='\0';
}
static void put_f32(uint8_t *b, size_t *o, float v) { memcpy(b+*o,&v,4); *o+=4; }
static void put_u16(uint8_t *b, size_t *o, uint16_t v){ memcpy(b+*o,&v,2); *o+=2; }

/* Build a minimal skinned glTF whose skin joints are nodes named jnames[0..nj-1]; verts influence
 * joints 0..nj-2 (last joint left uninfluenced ⇒ should get 0 probes). POINTS primitive (no triangle
 * count constraint). Returns 0 on success. */
static int build_skinned_gltf(const char *path, const char *const *jnames, int nj, int bind_skin, int bad_joint) {
    int V = (nj-1)*2;
    uint8_t bin[2048]; size_t off=0;
    size_t ibm_off = off; for (int i=0;i<nj;++i){ float I[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; for(int k=0;k<16;k++) put_f32(bin,&off,I[k]); }
    size_t pos_off = off; float mn[3]={1e9f,1e9f,1e9f}, mx[3]={-1e9f,-1e9f,-1e9f};
    for (int i=0;i<nj-1;++i) for (int s=0;s<2;++s){
        float P[3]={ (float)i + (s?0.1f:-0.1f), s?0.1f:-0.1f, s?0.1f:-0.1f };
        for (int c=0;c<3;c++){ put_f32(bin,&off,P[c]); if(P[c]<mn[c])mn[c]=P[c]; if(P[c]>mx[c])mx[c]=P[c]; }
    }
    /* JOINTS_0: vertex (i,s) → skin joint i; if bad_joint, vertex 0 references an out-of-range joint. */
    size_t jnt_off = off; { int first=1; for (int i=0;i<nj-1;++i) for (int s=0;s<2;++s){
        uint16_t ji = (bad_joint && first) ? (uint16_t)nj : (uint16_t)i; first=0;
        put_u16(bin,&off,ji); put_u16(bin,&off,0); put_u16(bin,&off,0); put_u16(bin,&off,0); } }
    size_t wgt_off = off; for (int i=0;i<nj-1;++i) for (int s=0;s<2;++s){ put_f32(bin,&off,1.0f); put_f32(bin,&off,0); put_f32(bin,&off,0); put_f32(bin,&off,0); }
    size_t total = off;

    char b64[3072]; base64(bin, total, b64);
    char nodes[1024]={0}; size_t np=0;
    for (int i=0;i<nj;++i) np += (size_t)snprintf(nodes+np, sizeof nodes-np, "%s{\"name\":\"%s\"}", i?",":"", jnames[i]);
    np += (size_t)snprintf(nodes+np, sizeof nodes-np, ",{\"name\":\"mesh\",\"mesh\":0%s}", bind_skin ? ",\"skin\":0" : "");
    char joints[256]={0}; size_t jp=0; for (int i=0;i<nj;++i) jp += (size_t)snprintf(joints+jp,sizeof joints-jp,"%s%d", i?",":"", i);
    char scene[256]={0}; size_t sp=0; for (int i=0;i<=nj;++i) sp += (size_t)snprintf(scene+sp,sizeof scene-sp,"%s%d", i?",":"", i);

    char json[6144];
    snprintf(json, sizeof json,
        "{\"asset\":{\"version\":\"2.0\"},\"scene\":0,\"scenes\":[{\"nodes\":[%s]}],"
        "\"nodes\":[%s],"
        "\"skins\":[{\"skeleton\":0,\"joints\":[%s],\"inverseBindMatrices\":0}],"
        "\"meshes\":[{\"primitives\":[{\"mode\":0,\"attributes\":{\"POSITION\":1,\"JOINTS_0\":2,\"WEIGHTS_0\":3}}]}],"
        "\"buffers\":[{\"byteLength\":%zu,\"uri\":\"data:application/octet-stream;base64,%s\"}],"
        "\"bufferViews\":["
          "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%d},"
          "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%d},"
          "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%d},"
          "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%d}],"
        "\"accessors\":["
          "{\"bufferView\":0,\"componentType\":5126,\"count\":%d,\"type\":\"MAT4\"},"
          "{\"bufferView\":1,\"componentType\":5126,\"count\":%d,\"type\":\"VEC3\",\"min\":[%g,%g,%g],\"max\":[%g,%g,%g]},"
          "{\"bufferView\":2,\"componentType\":5123,\"count\":%d,\"type\":\"VEC4\"},"
          "{\"bufferView\":3,\"componentType\":5126,\"count\":%d,\"type\":\"VEC4\"}]}",
        scene, nodes, joints, total, b64,
        ibm_off, nj*64, pos_off, V*12, jnt_off, V*8, wgt_off, V*16,
        nj, V, mn[0],mn[1],mn[2], mx[0],mx[1],mx[2], V, V);

    FILE *f = fopen(path, "wb"); if (!f) return -1;
    size_t w = fwrite(json, 1, strlen(json), f); int err = (w != strlen(json)); if (fclose(f)!=0) err=1;
    return err ? -1 : 0;
}

static int run_with_mesh(const char *rig, const char *mesh, int fallback, int *eligible) {
    mrw_bake_options opt = {0};
    opt.input_path = rig; opt.mesh_path = mesh; opt.bake_fps = 30.0f; opt.mesh_skin_index = -1;
    opt.allow_probe_fallback = fallback;
    uint8_t *obuf=NULL; size_t osize=0; mrw_bake_stats w; char d[256]={0};
    mrw_result r = mrw_bake_run(&opt, &obuf, &osize, eligible, &w, d, sizeof d);
    if (r == MRW_OK) { mrw_blob b; if (mrw_blob_open(obuf,(uint64_t)osize,&b)!=MRW_OK) r=MRW_E_FORMAT; }
    mrw_authoring_free(obuf);
    return (int)r;
}

static int test_mesh_probes(void) {
    uint32_t jc=4, sc=5;
    build_chain(jc, sc, -1, -1, 0.6f, g_parent, g_rest, g_ib, g_samp);
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { 30.0f, sc, 0, g_samp, NULL };
    const char *rig = MRW_BAKE_TESTDIR "/bake_meshrig.mrw";
    CHECK(write_clipset(rig, &skel, &clip, 1) == 0);

    /* matched skin (names j0..j3) ⇒ mesh probes resolve, rig bakes eligible (j3 uninfluenced ⇒ 0 probes) */
    const char *matched[4] = {"j0","j1","j2","j3"};
    const char *mesh_ok = MRW_BAKE_TESTDIR "/mesh_ok.gltf";
    CHECK(build_skinned_gltf(mesh_ok, matched, 4, 1, 0) == 0);
    int el = 0;
    CHECK(run_with_mesh(rig, mesh_ok, 0, &el) == MRW_OK);
    CHECK(el == 1);

    /* skin joint named 'jX' (not in skeleton) ⇒ strict error, but accepted with --allow-probe-fallback */
    const char *missing[4] = {"j0","j1","jX","j3"};
    const char *mesh_bad = MRW_BAKE_TESTDIR "/mesh_missing.gltf";
    CHECK(build_skinned_gltf(mesh_bad, missing, 4, 1, 0) == 0);
    el = 0;
    CHECK(run_with_mesh(rig, mesh_bad, 0, &el) == MRW_E_INCOMPATIBLE);   /* strict: no synthetic certify */
    CHECK(run_with_mesh(rig, mesh_bad, 1, &el) == MRW_OK);               /* fallback: box for uncovered j2 */
    CHECK(el == 1);

    /* duplicate joint name ⇒ strict error too */
    const char *dup[4] = {"j0","j1","j1","j3"};
    const char *mesh_dup = MRW_BAKE_TESTDIR "/mesh_dup.gltf";
    CHECK(build_skinned_gltf(mesh_dup, dup, 4, 1, 0) == 0);
    el = 0;
    CHECK(run_with_mesh(rig, mesh_dup, 0, &el) == MRW_E_INCOMPATIBLE);

    /* mesh with no primitive bound to the skin ⇒ no probes ⇒ error (no vacuous certification) */
    const char *mesh_nobind = MRW_BAKE_TESTDIR "/mesh_nobind.gltf";
    CHECK(build_skinned_gltf(mesh_nobind, matched, 4, /*bind_skin*/0, 0) == 0);
    el = 0;
    CHECK(run_with_mesh(rig, mesh_nobind, 0, &el) == MRW_E_INCOMPATIBLE);

    /* out-of-range JOINTS index (with nonzero weight) ⇒ malformed mesh, not silently dropped */
    const char *mesh_badidx = MRW_BAKE_TESTDIR "/mesh_badidx.gltf";
    CHECK(build_skinned_gltf(mesh_badidx, matched, 4, 1, /*bad_joint*/1) == 0);
    el = 0;
    CHECK(run_with_mesh(rig, mesh_badidx, 0, &el) == MRW_E_FORMAT);

    printf("  [bake] --mesh probes: matched resolves; missing/dup/no-bind/bad-index strict-error; fallback boxes\n");
    return 0;
}

/* ===================================================================== unprobed structural reject */
static int test_structural_reject(void) {
    /* A reflected REST scale makes bone 1's skinning palette det<0 (non-decomposable) at every frame.
     * Even UNPROBED (probe_counts all 0), a structural failure must reject the whole rig. */
    uint32_t jc=4, sc=4, fc=4;
    build_chain(jc, sc, -1, -1, 0.4f, g_parent, g_rest, g_ib, g_samp);
    g_rest[1*10+7] = -1.0f;                              /* reflect bone 1's x-scale */
    compute_bind_inverse(jc, g_parent, g_rest, g_ib);    /* recompute inverse-bind for the reflected rest */
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, NAMES };
    mrw_clip clip = { 30.0f, sc, 0, g_samp, NULL };
    uint8_t *buf=NULL; size_t size=0;
    CHECK(mrw_authoring_build(&skel,&clip,1,NULL,&buf,&size)==MRW_OK);
    mrw_blob blob; CHECK(mrw_blob_open(buf,(uint64_t)size,&blob)==MRW_OK);
    mrw_skeleton_view sv; mrw_clip_view cv;
    CHECK(mrw_blob_skeleton(&blob,&sv)==MRW_OK);
    for (uint32_t i=0;i<blob.section_count;++i){ uint32_t ty=0; mrw_blob_section_type(&blob,i,&ty); if(ty==MRW_SECTION_CLIP){mrw_clip_view_at(&blob,i,&cv);break;} }

    mrw_mem_req sreq, treq; CHECK(mrw_bake_clip_requirements(jc, fc, &sreq, &treq)==MRW_OK);
    void *scratch = mrw_authoring_alloc(sreq.size);
    uint16_t *tex = (uint16_t *)mrw_authoring_alloc(treq.size);
    CHECK(scratch && tex);
    uint32_t pc[MAXJ]; for (uint32_t j=0;j<jc;++j) pc[j]=0;   /* the reflected bone is UNPROBED */
    mrw_bake_stats st;
    CHECK(mrw_bake_clip(&sv,&cv,fc,pc,NULL,1e-3f,scratch,sreq.size,tex,treq.size,&st) == MRW_OK);
    CHECK(st.eligible == 0);
    CHECK(st.reason == MRW_BAKE_STRUCTURAL);

    /* argument validation: NULL required pointers and probes==NULL with a positive count ⇒ error */
    CHECK(mrw_bake_clip(NULL,&cv,fc,pc,NULL,1e-3f,scratch,sreq.size,tex,treq.size,&st) == MRW_E_FORMAT);
    uint32_t pc1[MAXJ]; for (uint32_t j=0;j<jc;++j) pc1[j]=8;
    CHECK(mrw_bake_clip(&sv,&cv,fc,pc1,NULL,1e-3f,scratch,sreq.size,tex,treq.size,&st) == MRW_E_FORMAT);

    mrw_authoring_free(scratch); mrw_authoring_free(tex); mrw_authoring_free(buf);
    printf("  [bake] structural: unprobed non-decomposable bone rejects rig; NULL args rejected\n");
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_eligible_parity();
    rc |= test_ineligible();
    rc |= test_frame_counts();
    rc |= test_bad_fps();
    rc |= test_tolerance_flip();
    rc |= test_determinism();
    rc |= test_buffer_contract();
    rc |= test_quantization_reject();
    rc |= test_structural_reject();
    rc |= test_mesh_probes();
    if (rc == 0) printf("test_bake: OK\n");
    return rc;
}
