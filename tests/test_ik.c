/* Analytic CPU IK. Two-bone reaches a reachable target (recompose via
 * mrw_local_to_model, re-measure) incl. a non-root chain under a ROTATED, UNIFORMLY-SCALED parent
 * (proves the solved-parent write-back); min/max reach clamps; the pole controls the bend-plane side
 * + pole-on-axis degeneracy; weight 0/½/1; an analytic right-triangle check; zero-length bone,
 * non-parent-linked (MRW_E_INCOMPATIBLE) and non-uniform-scale (MRW_E_UNSUPPORTED) rejections; and
 * the aim solver (axis-onto-target, up stabilizes roll, weight blend, target≈joint no-op). */
#include "test_util.h"
#include "mrw_build.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define MAXJ 8u
static const float IQ[4] = { 0,0,0,1 };

static uint16_t g_parent[MAXJ];
static float    g_rest[MAXJ*10];
static float    g_ib[MAXJ*12];
static const char *g_names[MAXJ] = { "a","b","c","d","e","f","g","h" };

static void joint(uint32_t j, int par, const float q[4],
                  float tx,float ty,float tz, float sx,float sy,float sz){
    g_parent[j] = (par<0)?0xFFFFu:(uint16_t)par;
    g_rest[j*10+0]=q[0]; g_rest[j*10+1]=q[1]; g_rest[j*10+2]=q[2]; g_rest[j*10+3]=q[3];
    g_rest[j*10+4]=tx; g_rest[j*10+5]=ty; g_rest[j*10+6]=tz;
    g_rest[j*10+7]=sx; g_rest[j*10+8]=sy; g_rest[j*10+9]=sz;
}
static void axis_angle(float ax,float ay,float az,float ang,float q[4]){
    float n=sqrtf(ax*ax+ay*ay+az*az); if(n<1e-6f){ax=0;ay=0;az=1;n=1;}
    float s=sinf(ang*0.5f); q[0]=ax/n*s; q[1]=ay/n*s; q[2]=az/n*s; q[3]=cosf(ang*0.5f);
}

/* Build the current rig (g_parent/g_rest) into a blob; fill `locals` from rest and `model` via the
 * local→model compose. Returns the blob buffer (free with mrw_free); leaves the skeleton view in *sv. */
static uint8_t *build_rig(uint32_t jc, mrw_blob *blob, mrw_skeleton_view *sv,
                          mrw_trs *locals, float *model){
    compute_bind_inverse(jc, g_parent, g_rest, g_ib);
    static float samp[MAXJ*10];
    memcpy(samp, g_rest, (size_t)jc*10*sizeof(float));   /* 1 static sample = rest */
    mrw_skel skel = { jc, g_parent, g_rest, g_ib, g_names };
    mrw_clip clip = { 30.0f, 1, 0, samp, NULL, 0 };
    uint8_t *buf=NULL; size_t sz = mrw_build(&skel,&clip,1,NULL,&buf);
    CHECK_EQ(mrw_blob_open(buf,sz,blob), MRW_OK);
    CHECK_EQ(mrw_blob_skeleton(blob,sv), MRW_OK);
    for (uint32_t j=0;j<jc;++j) CHECK_EQ(mrw_skeleton_rest_local(sv,j,&locals[j]), MRW_OK);
    CHECK_EQ(mrw_local_to_model(sv, locals, model, jc), MRW_OK);
    return buf;
}
static void mpos(const float *model, uint32_t j, float o[3]){
    o[0]=model[j*12+3]; o[1]=model[j*12+7]; o[2]=model[j*12+11];
}
static void mdir(const float *model, uint32_t j, const float v[3], float o[3]){
    const float *m = model + (size_t)j*12;
    o[0]=m[0]*v[0]+m[1]*v[1]+m[2]*v[2];
    o[1]=m[4]*v[0]+m[5]*v[1]+m[6]*v[2];
    o[2]=m[8]*v[0]+m[9]*v[1]+m[10]*v[2];
}
static float dist3(const float a[3], const float b[3]){
    float dx=a[0]-b[0],dy=a[1]-b[1],dz=a[2]-b[2]; return sqrtf(dx*dx+dy*dy+dz*dz);
}
static float len3(const float a[3]){ return sqrtf(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }

/* Build a straight chain root(0)->mid(1)->end(2) along +x with bone lengths l1,l2 (uniform scale 1). */
static uint8_t *build_arm(float l1, float l2, mrw_blob *blob, mrw_skeleton_view *sv,
                          mrw_trs *locals, float *model){
    joint(0, -1, IQ, 0,0,0, 1,1,1);
    joint(1,  0, IQ, l1,0,0, 1,1,1);
    joint(2,  1, IQ, l2,0,0, 1,1,1);
    return build_rig(3, blob, sv, locals, model);
}

/* ===================================================================== reach + analytic + clamps */
static void test_two_bone_reach(void){
    mrw_blob blob; mrw_skeleton_view sv; mrw_trs locals[MAXJ]; float model[MAXJ*12], model2[MAXJ*12];

    /* analytic right-triangle: l1=l2=1, target (1,1,0), pole +y ⇒ end=(1,1,0), mid=(0,1,0) */
    uint8_t *buf = build_arm(1.0f, 1.0f, &blob, &sv, locals, model);
    float target[3]={1,1,0}, pole[3]={1,10,0};
    CHECK_EQ(mrw_ik_two_bone(&sv, model, locals, 0,1,2, target, pole, 1.0f, 3), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv, locals, model2, 3), MRW_OK);
    float endp[3], midp[3]; mpos(model2,2,endp); mpos(model2,1,midp);
    CHECK(dist3(endp,target) < 1e-4f);
    float midexp[3]={0,1,0}; CHECK(dist3(midp,midexp) < 1e-4f);
    mrw_free(buf);

    /* max-reach clamp: target far away ⇒ fully extended toward it (‖end−A‖ = l1+l2, on the ray) */
    buf = build_arm(1.0f, 1.0f, &blob, &sv, locals, model);
    float A[3]; mpos(model,0,A);
    float far[3]={5,0,0};
    CHECK_EQ(mrw_ik_two_bone(&sv, model, locals, 0,1,2, far, pole, 1.0f, 3), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv, locals, model2, 3), MRW_OK);
    mpos(model2,2,endp);
    { float v[3]={endp[0]-A[0],endp[1]-A[1],endp[2]-A[2]}; CHECK_NEAR(len3(v), 2.0f, 1e-4); }
    { float exp[3]={2,0,0}; CHECK(dist3(endp,exp) < 1e-4f); }   /* on the +x ray to target */
    mrw_free(buf);

    /* min-reach clamp: unequal bones l1=2,l2=1 (|l1−l2|=1); target near root ⇒ ‖end−A‖ = 1 */
    buf = build_arm(2.0f, 1.0f, &blob, &sv, locals, model);
    mpos(model,0,A);
    float near[3]={0.1f,0,0};
    CHECK_EQ(mrw_ik_two_bone(&sv, model, locals, 0,1,2, near, pole, 1.0f, 3), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv, locals, model2, 3), MRW_OK);
    mpos(model2,2,endp);
    { float v[3]={endp[0]-A[0],endp[1]-A[1],endp[2]-A[2]}; CHECK_NEAR(len3(v), 1.0f, 1e-4); }
    mrw_free(buf);
    printf("  [ik] two-bone reach + analytic + min/max clamp ok\n");
}

/* Signed amount the solved mid bends TOWARD the pole: (mid−A) projected onto the unit bend
 * direction b̂ = normalize(reject(pole−A, û)), û = dir(target−A). Positive ⇒ mid is on the pole's
 * side of the root→target line (the bend is symmetric about that line, so an absolute axis sign is
 * the wrong discriminator - the two poles flip THIS, not necessarily a world coordinate). */
static float bend_toward_pole(const float mid[3], const float A[3],
                              const float target[3], const float pole[3]){
    float u[3]={target[0]-A[0],target[1]-A[1],target[2]-A[2]}; float un=len3(u);
    if (un>1e-6f){ u[0]/=un; u[1]/=un; u[2]/=un; }
    float p[3]={pole[0]-A[0],pole[1]-A[1],pole[2]-A[2]};
    float pd=p[0]*u[0]+p[1]*u[1]+p[2]*u[2];
    float b[3]={p[0]-pd*u[0],p[1]-pd*u[1],p[2]-pd*u[2]}; float bn=len3(b);
    if (bn>1e-6f){ b[0]/=bn; b[1]/=bn; b[2]/=bn; }
    float m[3]={mid[0]-A[0],mid[1]-A[1],mid[2]-A[2]};
    return m[0]*b[0]+m[1]*b[1]+m[2]*b[2];
}

/* ===================================================================== pole side + degeneracy */
static void test_pole(void){
    mrw_blob blob; mrw_skeleton_view sv; mrw_trs locals[MAXJ]; float model[MAXJ*12], model2[MAXJ*12];
    float target[3]={1,1,0}, A[3]={0,0,0};

    /* the two poles put the mid on opposite sides of the root→target line; both reach the target */
    uint8_t *buf = build_arm(1.0f,1.0f,&blob,&sv,locals,model);
    float polePos[3]={1,10,0};
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,2,target,polePos,1.0f,3), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv,locals,model2,3), MRW_OK);
    float midp[3], endp[3]; mpos(model2,1,midp); mpos(model2,2,endp);
    float side_plus = bend_toward_pole(midp, A, target, polePos);
    CHECK(side_plus > 0.3f); CHECK(dist3(endp,target)<1e-4f);
    mrw_free(buf);

    buf = build_arm(1.0f,1.0f,&blob,&sv,locals,model);
    float poleNeg[3]={1,-10,0};
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,2,target,poleNeg,1.0f,3), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv,locals,model2,3), MRW_OK);
    mpos(model2,1,midp); mpos(model2,2,endp);
    float side_minus = bend_toward_pole(midp, A, target, poleNeg);
    CHECK(side_minus > 0.3f); CHECK(dist3(endp,target)<1e-4f);   /* mid bends toward ITS pole */
    mrw_free(buf);

    /* pole ON the bone axis (P−A ∥ û) ⇒ documented fallback; still reaches the target, all finite */
    buf = build_arm(1.0f,1.0f,&blob,&sv,locals,model);
    float poleOn[3]={0.5f,0.5f,0};      /* along û=(1,1,0)/√2 from A=origin */
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,2,target,poleOn,1.0f,3), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv,locals,model2,3), MRW_OK);
    mpos(model2,2,endp);
    CHECK(dist3(endp,target)<1e-4f);
    for (int k=0;k<3;k++) CHECK(isfinite(endp[k]));
    mrw_free(buf);
    printf("  [ik] pole side (±) + on-axis degeneracy ok\n");
}

/* ===================================================================== weight blend */
static void test_weight(void){
    mrw_blob blob; mrw_skeleton_view sv; mrw_trs locals[MAXJ]; float model[MAXJ*12], model2[MAXJ*12];
    float target[3]={1,1,0}, pole[3]={1,10,0};

    /* weight 0 ⇒ exact no-op (end stays at the rest (2,0,0)) */
    uint8_t *buf = build_arm(1.0f,1.0f,&blob,&sv,locals,model);
    mrw_trs save0 = locals[0], save1 = locals[1];
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,2,target,pole,0.0f,3), MRW_OK);
    CHECK(memcmp(&save0, &locals[0], sizeof(mrw_trs))==0);
    CHECK(memcmp(&save1, &locals[1], sizeof(mrw_trs))==0);
    mrw_free(buf);

    /* measure the rest gap, then weight ½ moves closer but not all the way, weight 1 reaches */
    buf = build_arm(1.0f,1.0f,&blob,&sv,locals,model);
    float endp[3]; mpos(model,2,endp); float gap0 = dist3(endp,target);
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,2,target,pole,0.5f,3), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv,locals,model2,3), MRW_OK);
    mpos(model2,2,endp); float gap_half = dist3(endp,target);
    CHECK(gap_half < gap0); CHECK(gap_half > 1e-3f);
    mrw_free(buf);

    buf = build_arm(1.0f,1.0f,&blob,&sv,locals,model);
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,2,target,pole,1.0f,3), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv,locals,model2,3), MRW_OK);
    mpos(model2,2,endp); CHECK(dist3(endp,target) < 1e-4f);
    mrw_free(buf);
    printf("  [ik] weight 0/½/1 blend ok\n");
}

/* ===================================================================== non-root rotated/scaled parent */
static void test_parented(void){
    mrw_blob blob; mrw_skeleton_view sv; mrw_trs locals[MAXJ]; float model[MAXJ*12], model2[MAXJ*12];
    /* j0 pelvis: rotated 90° about Z, translated, UNIFORM scale 1.5; chain j1->j2->j3 below it */
    float qz[4]; axis_angle(0,0,1, 1.5707963f, qz);
    joint(0, -1, qz, 2.0f,1.0f,0.0f, 1.5f,1.5f,1.5f);
    joint(1,  0, IQ, 0.3f,0,0, 1,1,1);
    joint(2,  1, IQ, 1.0f,0,0, 1,1,1);
    joint(3,  2, IQ, 1.0f,0,0, 1,1,1);
    uint8_t *buf = build_rig(4, &blob, &sv, locals, model);

    float A[3], B[3], C[3]; mpos(model,1,A); mpos(model,2,B); mpos(model,3,C);
    float ba[3]={B[0]-A[0],B[1]-A[1],B[2]-A[2]}, cb[3]={C[0]-B[0],C[1]-B[1],C[2]-B[2]};
    float l1=len3(ba), l2=len3(cb);
    CHECK_NEAR(l1, 1.5f, 1e-4); CHECK_NEAR(l2, 1.5f, 1e-4);   /* world bone lengths scaled by 1.5 */

    /* reachable target near the root (‖T−A‖ < l1+l2), pole offset out of plane */
    float target[3]={A[0]+0.5f, A[1]+1.0f, A[2]+0.3f};
    float pole[3]={A[0], A[1], A[2]+5.0f};
    CHECK_EQ(mrw_ik_two_bone(&sv, model, locals, 1,2,3, target, pole, 1.0f, 4), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv, locals, model2, 4), MRW_OK);
    float endp[3]; mpos(model2,3,endp);
    CHECK(dist3(endp,target) < 1e-3f);    /* solved-parent write-back: end reaches the model-space target */

    /* root position is unchanged (IK rotates, never translates the chain) */
    float A2[3]; mpos(model2,1,A2); CHECK(dist3(A,A2) < 1e-5f);
    mrw_free(buf);
    printf("  [ik] non-root chain under rotated+scaled parent ok\n");
}

/* ===================================================================== rejections + degenerate bone */
static void test_rejections(void){
    mrw_blob blob; mrw_skeleton_view sv; mrw_trs locals[MAXJ]; float model[MAXJ*12];
    float target[3]={1,0.5f,0}, pole[3]={0,10,0};

    /* zero-length end bone (end coincident with mid) ⇒ no-op, locals unchanged */
    joint(0,-1,IQ,0,0,0,1,1,1); joint(1,0,IQ,1,0,0,1,1,1); joint(2,1,IQ,0,0,0,1,1,1);
    uint8_t *buf = build_rig(3,&blob,&sv,locals,model);
    mrw_trs s0=locals[0], s1=locals[1];
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,2,target,pole,1.0f,3), MRW_OK);
    CHECK(memcmp(&s0,&locals[0],sizeof(mrw_trs))==0);
    CHECK(memcmp(&s1,&locals[1],sizeof(mrw_trs))==0);
    mrw_free(buf);

    /* non-parent-linked triple ⇒ MRW_E_INCOMPATIBLE (4-joint chain, end_j not a child of mid_j) */
    joint(0,-1,IQ,0,0,0,1,1,1); joint(1,0,IQ,1,0,0,1,1,1);
    joint(2,1,IQ,1,0,0,1,1,1);  joint(3,2,IQ,1,0,0,1,1,1);
    buf = build_rig(4,&blob,&sv,locals,model);
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,3,target,pole,1.0f,4), MRW_E_INCOMPATIBLE);
    /* out-of-range joint index ⇒ RANGE; capacity */
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,9,target,pole,1.0f,4), MRW_E_RANGE);
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,2,target,pole,1.0f,2), MRW_E_CAPACITY);
    /* non-finite target ⇒ RANGE */
    { float bad[3]={1,(float)NAN,0}; CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,2,bad,pole,1.0f,4), MRW_E_RANGE); }
    mrw_free(buf);

    /* non-uniform scale on the root joint ⇒ MRW_E_UNSUPPORTED (similarity precondition violated) */
    joint(0,-1,IQ,0,0,0,1,1,1); joint(1,0,IQ,1,0,0,1,2,1); joint(2,1,IQ,1,0,0,1,1,1);
    buf = build_rig(3,&blob,&sv,locals,model);
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,2,target,pole,1.0f,3), MRW_E_UNSUPPORTED);
    mrw_free(buf);

    /* non-uniform scale on the END joint ⇒ MRW_E_UNSUPPORTED too (the WHOLE chain must be a similarity
     * - even though the solve only reads end_j's position) */
    joint(0,-1,IQ,0,0,0,1,1,1); joint(1,0,IQ,1,0,0,1,1,1); joint(2,1,IQ,1,0,0,2,1,1);
    buf = build_rig(3,&blob,&sv,locals,model);
    CHECK_EQ(mrw_ik_two_bone(&sv,model,locals,0,1,2,target,pole,1.0f,3), MRW_E_UNSUPPORTED);
    mrw_free(buf);
    printf("  [ik] zero-bone no-op / non-linked INCOMPATIBLE / non-uniform (root+end) UNSUPPORTED ok\n");
}

/* ===================================================================== aim / look-at */
static void test_aim(void){
    mrw_blob blob; mrw_skeleton_view sv; mrw_trs locals[MAXJ]; float model[MAXJ*12], model2[MAXJ*12];
    float aim[3]={0,0,1}, up[3]={0,1,0};   /* local +Z aims, +Y is up */

    /* single root joint at the origin: aim local +Z at +X, up +Y. World +Z ⇒ +X; world up ⇒ +Y. */
    joint(0,-1,IQ, 0,0,0, 1,1,1);
    uint8_t *buf = build_rig(1,&blob,&sv,locals,model);
    float target[3]={3,0,0}, upm[3]={0,1,0};
    CHECK_EQ(mrw_ik_aim(&sv, model, locals, 0, aim, up, target, upm, 1.0f, 1), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv, locals, model2, 1), MRW_OK);
    float waim[3]; mdir(model2,0,aim,waim);
    { float n=len3(waim); float d[3]={waim[0]/n,waim[1]/n,waim[2]/n}, exp[3]={1,0,0};
      CHECK(dist3(d,exp) < 1e-4f); }                 /* aim axis points at the target */
    float wup[3]; mdir(model2,0,up,wup);
    { float n=len3(wup); float d[3]={wup[0]/n,wup[1]/n,wup[2]/n}, exp[3]={0,1,0};
      CHECK(dist3(d,exp) < 1e-4f); }                 /* up stabilized toward up_model */
    mrw_free(buf);

    /* roll: same aim target, but up_model = +Z (perp to aim +X) ⇒ world up rolls toward +Z */
    joint(0,-1,IQ, 0,0,0, 1,1,1);
    buf = build_rig(1,&blob,&sv,locals,model);
    float upz[3]={0,0,1};
    CHECK_EQ(mrw_ik_aim(&sv, model, locals, 0, aim, up, target, upz, 1.0f, 1), MRW_OK);
    CHECK_EQ(mrw_local_to_model(&sv, locals, model2, 1), MRW_OK);
    mdir(model2,0,aim,waim); mdir(model2,0,up,wup);
    { float n=len3(waim); float d[3]={waim[0]/n,waim[1]/n,waim[2]/n}, exp[3]={1,0,0}; CHECK(dist3(d,exp)<1e-4f); }
    { float n=len3(wup);  float d[3]={wup[0]/n,wup[1]/n,wup[2]/n},  exp[3]={0,0,1}; CHECK(dist3(d,exp)<1e-4f); }
    mrw_free(buf);

    /* weight 0 ⇒ no-op; weight 1 ⇒ aimed (compare orientations) */
    joint(0,-1,IQ, 0,0,0, 1,1,1);
    buf = build_rig(1,&blob,&sv,locals,model);
    mrw_trs save = locals[0];
    CHECK_EQ(mrw_ik_aim(&sv, model, locals, 0, aim, up, target, upm, 0.0f, 1), MRW_OK);
    CHECK(memcmp(&save,&locals[0],sizeof(mrw_trs))==0);

    /* target ≈ joint ⇒ documented no-op */
    float here[3]={0,0,0};
    CHECK_EQ(mrw_ik_aim(&sv, model, locals, 0, aim, up, here, upm, 1.0f, 1), MRW_OK);
    CHECK(memcmp(&save,&locals[0],sizeof(mrw_trs))==0);

    /* collinear aim/up axes ⇒ RANGE; null + capacity + non-finite contract */
    float upc[3]={0,0,1};   /* collinear with aim +Z */
    CHECK_EQ(mrw_ik_aim(&sv, model, locals, 0, aim, upc, target, upm, 1.0f, 1), MRW_E_RANGE);
    CHECK_EQ(mrw_ik_aim(NULL, model, locals, 0, aim, up, target, upm, 1.0f, 1), MRW_E_RANGE);
    CHECK_EQ(mrw_ik_aim(&sv, model, locals, 9, aim, up, target, upm, 1.0f, 1), MRW_E_RANGE);
    CHECK_EQ(mrw_ik_aim(&sv, model, locals, 0, aim, up, target, upm, 1.0f, 0), MRW_E_CAPACITY);
    { float bad[3]={(float)INFINITY,0,0}; CHECK_EQ(mrw_ik_aim(&sv,model,locals,0,aim,up,bad,upm,1.0f,1), MRW_E_RANGE); }
    mrw_free(buf);
    printf("  [ik] aim axis-onto-target / roll / weight / no-op / contract ok\n");
}

int main(void){
    test_two_bone_reach();
    test_pole();
    test_weight();
    test_parented();
    test_rejections();
    test_aim();
    printf(g_fail ? "test_ik: %d FAILED\n" : "test_ik: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
