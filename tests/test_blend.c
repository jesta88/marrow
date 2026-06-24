/* Component-space pose blend + masking. Weight endpoints, midpoint symmetry,
 * unit-quat output, mask gating (0 / 1 / partial), exact-alias in-place == non-aliased, and the
 * ABI error contract (zero-count, capacity, non-finite incl. a NaN in mask, no partial write)
 * across tail counts. The single-pose blend is the reference the batch kernel is checked against. */
#include "test_util.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define MAXJ 16u

/* LCG RNG trio. */
typedef struct { uint32_t s; } rng;
static uint32_t ru32(rng *r){ r->s = r->s*1664525u + 1013904223u; return r->s; }
static float    rf(rng *r, float lo, float hi){ return lo + (hi-lo)*((float)(ru32(r)>>8)/(float)(1u<<24)); }
static void rquat(rng *r, float q[4]){
    float ax=rf(r,-1,1), ay=rf(r,-1,1), az=rf(r,-1,1);
    float n=sqrtf(ax*ax+ay*ay+az*az); if(n<1e-6f){ax=0;ay=0;az=1;n=1;}
    float ang=rf(r,-3.14159265f,3.14159265f), s=sinf(ang*0.5f);
    q[0]=ax/n*s; q[1]=ay/n*s; q[2]=az/n*s; q[3]=cosf(ang*0.5f);
}
static void rpose(rng *r, mrw_trs *p){
    rquat(r, p->rot);
    for (int k=0;k<3;k++){ p->trans[k]=rf(r,-2,2); p->scale[k]=rf(r,0.4f,2.0f); }
}
static void rposes(rng *r, mrw_trs *a, uint32_t n){ for (uint32_t j=0;j<n;j++) rpose(r,&a[j]); }

static int rot_near(const float a[4], const float b[4], double eps){
    double d=(double)a[0]*b[0]+(double)a[1]*b[1]+(double)a[2]*b[2]+(double)a[3]*b[3];
    return fabs(fabs(d)-1.0) <= eps;                 /* same rotation up to sign (q ≡ −q) */
}
static int is_unit(const float q[4], double eps){
    double n=sqrt((double)q[0]*q[0]+(double)q[1]*q[1]+(double)q[2]*q[2]+(double)q[3]*q[3]);
    return fabs(n-1.0) <= eps;
}

/* Independent reference: nlerp rot, lerp trans/scale at the effective weight. */
static void ref_blend_joint(const mrw_trs *a, const mrw_trs *b, float we, mrw_trs *o){
    float dot=a->rot[0]*b->rot[0]+a->rot[1]*b->rot[1]+a->rot[2]*b->rot[2]+a->rot[3]*b->rot[3];
    float s=(dot<0.0f)?-1.0f:1.0f, q[4];
    for (int k=0;k<4;k++) q[k]=a->rot[k]+we*(s*b->rot[k]-a->rot[k]);
    float n2=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3], inv=(n2>0)?1.0f/sqrtf(n2):0.0f;
    for (int k=0;k<4;k++) o->rot[k]=q[k]*inv;
    for (int k=0;k<3;k++){ o->trans[k]=a->trans[k]+we*(b->trans[k]-a->trans[k]);
                           o->scale[k]=a->scale[k]+we*(b->scale[k]-a->scale[k]); }
}
static void ref_blend(const mrw_trs *a, const mrw_trs *b, float w, const float *mask,
                      mrw_trs *out, uint32_t n){
    float wc = w<0?0:(w>1?1:w);
    for (uint32_t j=0;j<n;j++){
        float mj = mask ? (mask[j]<0?0:(mask[j]>1?1:mask[j])) : 1.0f;
        float we = wc*mj; we = we<0?0:(we>1?1:we);
        ref_blend_joint(&a[j], &b[j], we, &out[j]);
    }
}

/* ===================================================================== weights + symmetry + unit */
static void test_weights(void){
    static mrw_trs a[MAXJ], b[MAXJ], out[MAXJ], out2[MAXJ];
    const uint32_t tails[] = { 0,1,3,7,8,9,16 };
    rng r = { 0xB1E2D01u };
    for (size_t ti=0; ti<sizeof tails/sizeof tails[0]; ++ti){
        uint32_t n = tails[ti];
        rposes(&r, a, n>0?n:1); rposes(&r, b, n>0?n:1);

        /* zero-count is a clean no-op regardless of pointers */
        if (n==0){ CHECK_EQ(mrw_pose_blend(a,b,0.5f,NULL,out,0,MAXJ), MRW_OK); continue; }

        /* w=0 ⇒ out=a (trans/scale exact via mul-by-zero; rot = a as a rotation) */
        CHECK_EQ(mrw_pose_blend(a,b,0.0f,NULL,out,n,MAXJ), MRW_OK);
        for (uint32_t j=0;j<n;j++){
            CHECK(rot_near(out[j].rot, a[j].rot, 1e-6));
            for (int k=0;k<3;k++){ CHECK(out[j].trans[k]==a[j].trans[k]); CHECK(out[j].scale[k]==a[j].scale[k]); }
            CHECK(is_unit(out[j].rot, 1e-6));
        }
        /* w=1 ⇒ trans/scale = b; rot = b as a rotation */
        CHECK_EQ(mrw_pose_blend(a,b,1.0f,NULL,out,n,MAXJ), MRW_OK);
        for (uint32_t j=0;j<n;j++){
            CHECK(rot_near(out[j].rot, b[j].rot, 1e-6));
            for (int k=0;k<3;k++){ CHECK_NEAR(out[j].trans[k], b[j].trans[k], 1e-6);
                                   CHECK_NEAR(out[j].scale[k], b[j].scale[k], 1e-6); }
            CHECK(is_unit(out[j].rot, 1e-6));
        }
        /* clamp: w<0 ≡ w=0, w>1 ≡ w=1 */
        CHECK_EQ(mrw_pose_blend(a,b,-3.0f,NULL,out,n,MAXJ), MRW_OK);
        for (uint32_t j=0;j<n;j++) for (int k=0;k<3;k++) CHECK(out[j].trans[k]==a[j].trans[k]);
        CHECK_EQ(mrw_pose_blend(a,b,5.0f,NULL,out2,n,MAXJ), MRW_OK);
        for (uint32_t j=0;j<n;j++) for (int k=0;k<3;k++) CHECK_NEAR(out2[j].trans[k], b[j].trans[k], 1e-6);

        /* random weight: matches the reference formula; unit quats; midpoint symmetry */
        for (int it=0; it<8; ++it){
            float w = rf(&r, 0.0f, 1.0f);
            CHECK_EQ(mrw_pose_blend(a,b,w,NULL,out,n,MAXJ), MRW_OK);
            ref_blend(a,b,w,NULL,out2,n);
            for (uint32_t j=0;j<n;j++){
                CHECK(is_unit(out[j].rot, 1e-6));
                CHECK(rot_near(out[j].rot, out2[j].rot, 1e-6));
                for (int k=0;k<3;k++){ CHECK_NEAR(out[j].trans[k], out2[j].trans[k], 1e-6);
                                       CHECK_NEAR(out[j].scale[k], out2[j].scale[k], 1e-6); }
            }
            /* blend(a,b,w) ≡ blend(b,a,1−w) as rotation + componentwise trans/scale */
            mrw_trs sym[MAXJ];
            CHECK_EQ(mrw_pose_blend(b,a,1.0f-w,NULL,sym,n,MAXJ), MRW_OK);
            for (uint32_t j=0;j<n;j++){
                CHECK(rot_near(out[j].rot, sym[j].rot, 1e-5));
                for (int k=0;k<3;k++){ CHECK_NEAR(out[j].trans[k], sym[j].trans[k], 1e-5);
                                       CHECK_NEAR(out[j].scale[k], sym[j].scale[k], 1e-5); }
            }
        }
    }
    printf("  [blend] weights/symmetry/unit ok\n");
}

/* ===================================================================== mask gating */
static void test_mask(void){
    static mrw_trs a[MAXJ], b[MAXJ], out[MAXJ], ref[MAXJ];
    uint32_t n = 9;
    rng r = { 0x5A5Au };
    rposes(&r, a, n); rposes(&r, b, n);
    float mask[MAXJ];
    /* mix of 0 / 1 / partial / out-of-range (clamped) */
    float vals[9] = { 0.0f, 1.0f, 0.5f, 0.25f, -1.0f, 2.0f, 0.0f, 0.75f, 1.0f };
    for (uint32_t j=0;j<n;j++) mask[j]=vals[j];
    float w = 0.6f;
    CHECK_EQ(mrw_pose_blend(a,b,w,mask,out,n,MAXJ), MRW_OK);
    ref_blend(a,b,w,mask,ref,n);
    for (uint32_t j=0;j<n;j++){
        CHECK(rot_near(out[j].rot, ref[j].rot, 1e-6));
        for (int k=0;k<3;k++){ CHECK_NEAR(out[j].trans[k], ref[j].trans[k], 1e-6);
                               CHECK_NEAR(out[j].scale[k], ref[j].scale[k], 1e-6); }
    }
    /* masked-out (mask=0) joints stay exactly a (trans/scale exact, rot as rotation) */
    for (uint32_t j=0;j<n;j++) if (vals[j]==0.0f){
        for (int k=0;k<3;k++){ CHECK(out[j].trans[k]==a[j].trans[k]); CHECK(out[j].scale[k]==a[j].scale[k]); }
        CHECK(rot_near(out[j].rot, a[j].rot, 1e-6));
    }
    /* mask=1 joints equal the unmasked blend at w */
    mrw_trs full[MAXJ]; CHECK_EQ(mrw_pose_blend(a,b,w,NULL,full,n,MAXJ), MRW_OK);
    for (uint32_t j=0;j<n;j++) if (vals[j]==1.0f)
        for (int k=0;k<3;k++) CHECK_NEAR(out[j].trans[k], full[j].trans[k], 1e-6);
    printf("  [blend] mask 0/1/partial/clamped ok\n");
}

/* ===================================================================== exact-alias in-place */
static void test_alias(void){
    static mrw_trs a[MAXJ], b[MAXJ], ref[MAXJ], buf[MAXJ];
    uint32_t n = 8;
    rng r = { 0xA11A5u };
    rposes(&r, a, n); rposes(&r, b, n);
    float w = 0.37f;
    ref_blend(a,b,w,NULL,ref,n);

    /* out == a (same buffer): copy a into buf, blend(buf,b,..,buf) */
    memcpy(buf, a, n*sizeof(mrw_trs));
    CHECK_EQ(mrw_pose_blend(buf,b,w,NULL,buf,n,MAXJ), MRW_OK);
    for (uint32_t j=0;j<n;j++){ CHECK(rot_near(buf[j].rot, ref[j].rot, 1e-6));
        for (int k=0;k<3;k++){ CHECK(buf[j].trans[k]==ref[j].trans[k]); CHECK(buf[j].scale[k]==ref[j].scale[k]); } }

    /* out == b (same buffer): copy b into buf, blend(a,buf,..,buf) */
    memcpy(buf, b, n*sizeof(mrw_trs));
    CHECK_EQ(mrw_pose_blend(a,buf,w,NULL,buf,n,MAXJ), MRW_OK);
    for (uint32_t j=0;j<n;j++){ CHECK(rot_near(buf[j].rot, ref[j].rot, 1e-6));
        for (int k=0;k<3;k++){ CHECK(buf[j].trans[k]==ref[j].trans[k]); CHECK(buf[j].scale[k]==ref[j].scale[k]); } }
    printf("  [blend] exact-alias in-place == non-aliased ok\n");
}

/* ===================================================================== ABI error contract */
static void test_contract(void){
    static mrw_trs a[MAXJ], b[MAXJ], out[MAXJ];
    uint32_t n = 8;
    rng r = { 0xC0FFEEu };
    rposes(&r, a, n); rposes(&r, b, n);

    /* NULL handles ⇒ RANGE */
    CHECK_EQ(mrw_pose_blend(NULL,b,0.5f,NULL,out,n,MAXJ), MRW_E_RANGE);
    CHECK_EQ(mrw_pose_blend(a,NULL,0.5f,NULL,out,n,MAXJ), MRW_E_RANGE);
    CHECK_EQ(mrw_pose_blend(a,b,0.5f,NULL,NULL,n,MAXJ), MRW_E_RANGE);
    /* capacity */
    CHECK_EQ(mrw_pose_blend(a,b,0.5f,NULL,out,MAXJ+1,MAXJ), MRW_E_CAPACITY);
    /* zero-count no-op (pointers ignored) */
    CHECK_EQ(mrw_pose_blend(NULL,NULL,0.5f,NULL,NULL,0,MAXJ), MRW_OK);

    /* non-finite ⇒ RANGE with NO partial write (canary the whole output) */
    mrw_trs save = a[3];
    a[3].rot[1] = (float)NAN;
    memset(out, 0xAB, n*sizeof(mrw_trs));
    unsigned char canary[sizeof(mrw_trs)*MAXJ]; memset(canary, 0xAB, sizeof canary);
    CHECK_EQ(mrw_pose_blend(a,b,0.5f,NULL,out,n,MAXJ), MRW_E_RANGE);
    CHECK(memcmp(out, canary, n*sizeof(mrw_trs))==0);
    a[3] = save;
    /* a NaN in mask ⇒ RANGE (mask is finite-validated) */
    float mask[MAXJ]; for (uint32_t j=0;j<n;j++) mask[j]=0.5f; mask[5]=(float)INFINITY;
    CHECK_EQ(mrw_pose_blend(a,b,0.5f,mask,out,n,MAXJ), MRW_E_RANGE);
    /* non-finite weight ⇒ RANGE */
    CHECK_EQ(mrw_pose_blend(a,b,(float)NAN,NULL,out,n,MAXJ), MRW_E_RANGE);
    printf("  [blend] ABI contract (null/capacity/zero/non-finite/no-partial-write) ok\n");
}

int main(void){
    test_weights();
    test_mask();
    test_alias();
    test_contract();
    printf(g_fail ? "test_blend: %d FAILED\n" : "test_blend: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
