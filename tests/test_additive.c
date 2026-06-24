/* Additive pose (make + accumulate). The rotation-space round-trip
 * (compare rotations / probe points, NOT raw quaternion components), incl. deltas with delta.w<0 and
 * the same-rotation/opposite-sign pair (canon ⇒ no −identity spin); exact trans/scale round-trip;
 * ratio-scale identity (identity delta == identity TRS); w=0 / identity-delta ⇒ base; mask gating;
 * the non-positive/zero base-scale guard; and the ABI error contract. */
#include "test_util.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define MAXJ 16u

typedef struct { uint32_t s; } rng;
static uint32_t ru32(rng *r){ r->s = r->s*1664525u + 1013904223u; return r->s; }
static float    rf(rng *r, float lo, float hi){ return lo + (hi-lo)*((float)(ru32(r)>>8)/(float)(1u<<24)); }
static void rquat(rng *r, float q[4]){
    float ax=rf(r,-1,1), ay=rf(r,-1,1), az=rf(r,-1,1);
    float n=sqrtf(ax*ax+ay*ay+az*az); if(n<1e-6f){ax=0;ay=0;az=1;n=1;}
    float ang=rf(r,-3.14159265f,3.14159265f), s=sinf(ang*0.5f);
    q[0]=ax/n*s; q[1]=ay/n*s; q[2]=az/n*s; q[3]=cosf(ang*0.5f);
}
static void axis_angle(float ax,float ay,float az,float ang,float q[4]){
    float n=sqrtf(ax*ax+ay*ay+az*az); if(n<1e-6f){ax=0;ay=0;az=1;n=1;}
    float s=sinf(ang*0.5f); q[0]=ax/n*s; q[1]=ay/n*s; q[2]=az/n*s; q[3]=cosf(ang*0.5f);
}
static void rpose(rng *r, mrw_trs *p){
    rquat(r,p->rot); for (int k=0;k<3;k++){ p->trans[k]=rf(r,-2,2); p->scale[k]=rf(r,0.4f,2.0f); }
}

static int rot_near(const float a[4], const float b[4], double eps){
    double d=(double)a[0]*b[0]+(double)a[1]*b[1]+(double)a[2]*b[2]+(double)a[3]*b[3];
    return fabs(fabs(d)-1.0) <= eps;
}
/* worst probe-point displacement between two full TRS transforms (rot+trans+scale). */
static float probe_dist(const mrw_trs *x, const mrw_trs *y){
    float ax[12], ay[12]; mrw_trs_to_affine(x,ax); mrw_trs_to_affine(y,ay);
    float worst=0.0f;
    for (int sx=-1;sx<=1;sx+=2) for (int sy=-1;sy<=1;sy+=2) for (int sz=-1;sz<=1;sz+=2){
        float p[3]={0.2f*sx,0.2f*sy,0.2f*sz}, pa[3], pb[3];
        pa[0]=ax[0]*p[0]+ax[1]*p[1]+ax[2]*p[2]+ax[3];
        pa[1]=ax[4]*p[0]+ax[5]*p[1]+ax[6]*p[2]+ax[7];
        pa[2]=ax[8]*p[0]+ax[9]*p[1]+ax[10]*p[2]+ax[11];
        pb[0]=ay[0]*p[0]+ay[1]*p[1]+ay[2]*p[2]+ay[3];
        pb[1]=ay[4]*p[0]+ay[5]*p[1]+ay[6]*p[2]+ay[7];
        pb[2]=ay[8]*p[0]+ay[9]*p[1]+ay[10]*p[2]+ay[11];
        float dx=pa[0]-pb[0],dy=pa[1]-pb[1],dz=pa[2]-pb[2], d=sqrtf(dx*dx+dy*dy+dz*dz);
        if (d>worst) worst=d;
    }
    return worst;
}

/* ===================================================================== round-trip */
static void test_roundtrip(void){
    static mrw_trs base[MAXJ], pose[MAXJ], delta[MAXJ], out[MAXJ];
    const uint32_t tails[] = { 1,3,7,8,9,16 };
    rng r = { 0xADD17u };
    int saw_negw = 0;
    for (size_t ti=0; ti<sizeof tails/sizeof tails[0]; ++ti){
        uint32_t n = tails[ti];
        for (uint32_t j=0;j<n;j++){ rpose(&r,&base[j]); rpose(&r,&pose[j]); }
        CHECK_EQ(mrw_pose_make_additive(pose,base,delta,n,MAXJ), MRW_OK);
        for (uint32_t j=0;j<n;j++){
            CHECK(delta[j].rot[3] >= 0.0f);                 /* canon ⇒ w ≥ 0 */
            /* detect that some relative rotations exceed 180° (raw delta.w would be <0 pre-canon) */
            float dot=base[j].rot[0]*pose[j].rot[0]+base[j].rot[1]*pose[j].rot[1]
                     +base[j].rot[2]*pose[j].rot[2]+base[j].rot[3]*pose[j].rot[3];
            if (fabsf(dot) < 0.4f) saw_negw = 1;
        }
        CHECK_EQ(mrw_pose_accumulate(base,delta,1.0f,NULL,out,n,MAXJ), MRW_OK);
        for (uint32_t j=0;j<n;j++){
            CHECK(rot_near(out[j].rot, pose[j].rot, 1e-5));  /* rotation up to sign */
            for (int k=0;k<3;k++){ CHECK_NEAR(out[j].trans[k], pose[j].trans[k], 1e-5);  /* trans exact */
                                   CHECK_NEAR(out[j].scale[k], pose[j].scale[k], 1e-5); } /* scale exact (pos ratio) */
            CHECK(probe_dist(&out[j], &pose[j]) < 1e-4f);    /* rotation-space probe check */
        }
    }
    CHECK(saw_negw);   /* the random set includes >180° relative rotations (pre-canon delta.w<0) */
    printf("  [additive] round-trip (rot/trans/scale/probe; incl. delta.w<0) ok\n");
}

/* ===================================================================== explicit delta.w<0 + opposite sign */
static void test_canon(void){
    mrw_trs base, pose, delta, out;
    /* (1) explicit >180° relative rotation: base identity-rot, pose at 250° ⇒ raw delta.w<0 */
    base.rot[0]=base.rot[1]=base.rot[2]=0; base.rot[3]=1;
    for (int k=0;k<3;k++){ base.trans[k]=0.1f*(k+1); base.scale[k]=1.0f+0.1f*k; }
    axis_angle(0.3f,0.6f,0.7f, 250.0f*3.14159265f/180.0f, pose.rot);
    for (int k=0;k<3;k++){ pose.trans[k]=0.2f*(k+1); pose.scale[k]=1.2f-0.1f*k; }
    CHECK_EQ(mrw_pose_make_additive(&pose,&base,&delta,1,1), MRW_OK);
    CHECK(delta.rot[3] >= 0.0f);
    CHECK_EQ(mrw_pose_accumulate(&base,&delta,1.0f,NULL,&out,1,1), MRW_OK);
    CHECK(rot_near(out.rot, pose.rot, 1e-5));
    CHECK(probe_dist(&out,&pose) < 1e-4f);

    /* (2) same rotation, opposite quaternion sign: pose.rot = −base.rot ⇒ delta canonicalizes to
     *     identity, so accumulate at PARTIAL weight gives base.rot (no spurious 180° spin). */
    rng r = { 0x9911u }; rpose(&r,&base);
    pose = base; for (int k=0;k<4;k++) pose.rot[k] = -base.rot[k];
    CHECK_EQ(mrw_pose_make_additive(&pose,&base,&delta,1,1), MRW_OK);
    CHECK(rot_near(delta.rot, (float[4]){0,0,0,1}, 1e-5));   /* delta == identity rotation */
    for (float w=0.0f; w<=1.0f+1e-6f; w+=0.25f){
        CHECK_EQ(mrw_pose_accumulate(&base,&delta,w,NULL,&out,1,1), MRW_OK);
        CHECK(rot_near(out.rot, base.rot, 1e-5));            /* no spin at any weight */
    }
    printf("  [additive] canon: delta.w<0 round-trip + opposite-sign no-spin ok\n");
}

/* ===================================================================== identity delta / w=0 / mask */
static void test_identity_and_weight(void){
    static mrw_trs base[MAXJ], pose[MAXJ], delta[MAXJ], out[MAXJ];
    uint32_t n = 9;
    rng r = { 0x1DE17u };
    for (uint32_t j=0;j<n;j++){ rpose(&r,&base[j]); rpose(&r,&pose[j]); }

    /* identity delta == identity TRS: make_additive(X,X) ⇒ trans 0 exact, scale 1 exact, rot identity */
    CHECK_EQ(mrw_pose_make_additive(base,base,delta,n,MAXJ), MRW_OK);
    for (uint32_t j=0;j<n;j++){
        for (int k=0;k<3;k++){ CHECK(delta[j].trans[k]==0.0f); CHECK(delta[j].scale[k]==1.0f); }
        CHECK(rot_near(delta[j].rot, (float[4]){0,0,0,1}, 1e-6));
    }
    /* accumulate(base, identity_delta, w) == base for any w (exact trans/scale, rot as rotation) */
    for (float w=0.0f; w<=1.0f+1e-6f; w+=0.5f){
        CHECK_EQ(mrw_pose_accumulate(base,delta,w,NULL,out,n,MAXJ), MRW_OK);
        for (uint32_t j=0;j<n;j++){
            for (int k=0;k<3;k++){ CHECK(out[j].trans[k]==base[j].trans[k]); CHECK(out[j].scale[k]==base[j].scale[k]); }
            CHECK(rot_near(out[j].rot, base[j].rot, 1e-6));
        }
    }

    /* w=0 ⇒ base exactly (real delta): trans/scale exact (mul-by-zero), rot = base as rotation */
    CHECK_EQ(mrw_pose_make_additive(pose,base,delta,n,MAXJ), MRW_OK);
    CHECK_EQ(mrw_pose_accumulate(base,delta,0.0f,NULL,out,n,MAXJ), MRW_OK);
    for (uint32_t j=0;j<n;j++){
        for (int k=0;k<3;k++){ CHECK(out[j].trans[k]==base[j].trans[k]); CHECK(out[j].scale[k]==base[j].scale[k]); }
        CHECK(rot_near(out[j].rot, base[j].rot, 1e-6));
    }

    /* mask gating: mask=0 ⇒ base; mask=1 ⇒ full accumulate; partial ⇒ between */
    float mask[MAXJ]; for (uint32_t j=0;j<n;j++) mask[j] = (j%3==0)?0.0f:(j%3==1)?1.0f:0.5f;
    CHECK_EQ(mrw_pose_accumulate(base,delta,1.0f,mask,out,n,MAXJ), MRW_OK);
    mrw_trs full[MAXJ]; CHECK_EQ(mrw_pose_accumulate(base,delta,1.0f,NULL,full,n,MAXJ), MRW_OK);
    for (uint32_t j=0;j<n;j++){
        if (j%3==0){ for (int k=0;k<3;k++){ CHECK(out[j].trans[k]==base[j].trans[k]); CHECK(out[j].scale[k]==base[j].scale[k]); }
                     CHECK(rot_near(out[j].rot, base[j].rot, 1e-6)); }
        else if (j%3==1){ for (int k=0;k<3;k++) CHECK_NEAR(out[j].trans[k], full[j].trans[k], 1e-6);
                          CHECK(rot_near(out[j].rot, full[j].rot, 1e-6)); }
    }
    printf("  [additive] identity-delta / w=0 / mask gating ok\n");
}

/* ===================================================================== scale guard + contract */
static void test_scale_guard_and_contract(void){
    mrw_trs base, pose, delta, out; (void)out;
    base.rot[0]=base.rot[1]=base.rot[2]=0; base.rot[3]=1; pose=base;
    for (int k=0;k<3;k++){ base.trans[k]=0; pose.trans[k]=0; }
    /* zero / negative / overflowing base or ratio ⇒ delta.scale = 1 (no delta, scale guard) */
    base.scale[0]=0.0f;    pose.scale[0]=2.0f;     /* base ≤ 0 */
    base.scale[1]=-1.5f;   pose.scale[1]=2.0f;     /* base < 0 */
    base.scale[2]=1e-38f;  pose.scale[2]=1e38f;    /* ratio overflows ⇒ non-finite */
    CHECK_EQ(mrw_pose_make_additive(&pose,&base,&delta,1,1), MRW_OK);
    for (int k=0;k<3;k++) CHECK(delta.scale[k]==1.0f);

    /* a normal positive channel still round-trips */
    base.scale[0]=1.5f; pose.scale[0]=0.75f;
    CHECK_EQ(mrw_pose_make_additive(&pose,&base,&delta,1,1), MRW_OK);
    CHECK_NEAR(delta.scale[0], 0.5f, 1e-6);

    /* ABI contract: NULL ⇒ RANGE; capacity; zero-count no-op; non-finite ⇒ RANGE, no partial write */
    static mrw_trs a[MAXJ], b[MAXJ], o[MAXJ];
    for (uint32_t j=0;j<MAXJ;j++){ a[j]=base; b[j]=pose; }
    CHECK_EQ(mrw_pose_make_additive(NULL,b,o,4,MAXJ), MRW_E_RANGE);
    CHECK_EQ(mrw_pose_make_additive(a,b,NULL,4,MAXJ), MRW_E_RANGE);
    CHECK_EQ(mrw_pose_make_additive(a,b,o,MAXJ+1,MAXJ), MRW_E_CAPACITY);
    CHECK_EQ(mrw_pose_make_additive(a,b,o,0,MAXJ), MRW_OK);
    CHECK_EQ(mrw_pose_accumulate(NULL,b,1.0f,NULL,o,4,MAXJ), MRW_E_RANGE);
    CHECK_EQ(mrw_pose_accumulate(a,NULL,1.0f,NULL,o,4,MAXJ), MRW_E_RANGE);
    CHECK_EQ(mrw_pose_accumulate(a,b,1.0f,NULL,o,MAXJ+1,MAXJ), MRW_E_CAPACITY);
    CHECK_EQ(mrw_pose_accumulate(a,b,(float)NAN,NULL,o,4,MAXJ), MRW_E_RANGE);

    a[2].trans[0] = (float)INFINITY;
    unsigned char canary[sizeof(mrw_trs)*MAXJ]; memset(canary,0xAB,sizeof canary);
    memset(o,0xAB,4*sizeof(mrw_trs));
    CHECK_EQ(mrw_pose_make_additive(a,b,o,4,MAXJ), MRW_E_RANGE);   /* pose[2] non-finite */
    CHECK(memcmp(o,canary,4*sizeof(mrw_trs))==0);
    printf("  [additive] scale guard + ABI contract ok\n");
}

int main(void){
    test_roundtrip();
    test_canon();
    test_identity_and_weight();
    test_scale_guard_and_contract();
    printf(g_fail ? "test_additive: %d FAILED\n" : "test_additive: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
