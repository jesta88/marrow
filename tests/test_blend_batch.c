/* Across-instance batch blend + accumulate. Randomized scalar-vs-SSE2-vs-AVX2 parity
 * against the LOOPED single-pose reference (mrw_pose_blend / mrw_pose_accumulate + compose +
 * palette) across every tail count, f32 + f16 output, with/without a shared per-joint mask;
 * scalar is bit-exact, SIMD within tolerance; plus the f16-is-the-narrowed-f32 consistency
 * gate, an output canary, and the dispatch/incompatibility error contract. */
#include "bench_rig.h"
#include <stdlib.h>
#include <math.h>

static const uint32_t TAILS[] = { 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64,100 };

/* SIMD vs scalar-reference bound. Looser atol than the clip kernel's 1e-4 (test_batch): the
 * accumulate adds a per-joint quat-multiply + renormalize on top of the clip op-chain, and that
 * extra FMA/reassociation noise compounds through the deep (65-joint) hierarchy compose. Still
 * ~1000× tighter than any logic bug (a wrong lane/index/sign yields O(1) error). Scalar stays
 * bit-exact (exact f32 compare; f16 within ½ a binary16 ULP of the reference). */
#define SIMD_ATOL 5e-4
#define SIMD_RTOL 1e-3

static double f16_half_ulp(double v){
    int e; (void)frexp(fabs(v), &e);
    int be = e - 1; if (be < -14) be = -14;
    return 0.5 * ldexp(1.0, be - 10);
}

typedef struct { mrw_dispatch d; const char *name; int simd; } backend_t;
static int host_backends(backend_t *bks){
    int n = 0; mrw_dispatch t;
    mrw_dispatch_scalar(&t);              bks[n++] = (backend_t){ t, "scalar", 0 };
    if (mrw_dispatch_sse2(&t) == MRW_OK)  bks[n++] = (backend_t){ t, "sse2",   1 };
    if (mrw_dispatch_avx2(&t) == MRW_OK){
        bks[n++] = (backend_t){ t, "avx2", 1 };
        if (t.features & MRW_FEAT_FMA){ mrw_dispatch nf=t; nf.features &= ~(uint32_t)MRW_FEAT_FMA;
            bks[n++] = (backend_t){ nf, "avx2-nofma", 1 }; }
        if (t.features & MRW_FEAT_F16C){ mrw_dispatch n16=t; n16.features &= ~(uint32_t)MRW_FEAT_F16C;
            bks[n++] = (backend_t){ n16, "avx2-nof16c", 1 }; }
    }
    return n;
}

/* Build a {skeleton, clipA, clipB} blob (two clips on one skeleton). Free *buf with mrw_free. */
static size_t build_2clip(uint32_t jc, uint32_t sc, uint32_t flags, uint32_t seed, uint8_t **buf){
    static uint16_t parent[RIG_MAX_JOINTS];
    static float    rest[RIG_MAX_JOINTS*10], ib[RIG_MAX_JOINTS*12];
    static char     nb[RIG_MAX_JOINTS][8]; static const char *names[RIG_MAX_JOINTS];
    static float    sA[RIG_MAX_JOINTS*RIG_MAX_SAMPLES*10], sB[RIG_MAX_JOINTS*RIG_MAX_SAMPLES*10];
    rig_rng r = { seed ? seed : 1u };
    for (uint32_t j=0;j<jc;j++){
        parent[j] = (j==0)?0xFFFFu:(uint16_t)(j-1);
        float q[4]; rig_quat(&r,q);
        rest[j*10+0]=q[0]; rest[j*10+1]=q[1]; rest[j*10+2]=q[2]; rest[j*10+3]=q[3];
        rest[j*10+4]=(j==0)?0.0f:rig_f(&r,0.2f,1.2f);
        rest[j*10+5]=rig_f(&r,-0.3f,0.3f); rest[j*10+6]=rig_f(&r,-0.3f,0.3f);
        rest[j*10+7]=rest[j*10+8]=rest[j*10+9]=1.0f;
        snprintf(nb[j], sizeof nb[j], "j%u", j); names[j]=nb[j];
    }
    compute_bind_inverse(jc, parent, rest, ib);
    for (uint32_t j=0;j<jc;j++) for (uint32_t s=0;s<sc;s++){
        float *a=sA+((size_t)j*sc+s)*10, *b=sB+((size_t)j*sc+s)*10;
        float qa[4]; rig_quat(&r,qa); for(int k=0;k<4;k++) a[k]=qa[k];
        a[4]=rig_f(&r,-1,1); a[5]=rig_f(&r,-1,1); a[6]=rig_f(&r,-1,1); a[7]=a[8]=a[9]=1.0f;
        float qb[4]; rig_quat(&r,qb); for(int k=0;k<4;k++) b[k]=qb[k];
        b[4]=rig_f(&r,-1,1); b[5]=rig_f(&r,-1,1); b[6]=rig_f(&r,-1,1); b[7]=b[8]=b[9]=1.0f;
    }
    mrw_skel skel = { jc, parent, rest, ib, names };
    mrw_clip clips[2] = { { RIG_FPS, sc, flags, sA, NULL, 0 }, { RIG_FPS, sc, flags, sB, NULL, 0 } };
    return mrw_build(&skel, clips, 2, NULL, buf);
}

/* ---- per-instance single-pose reference (blend) ---- */
static void oracle_blend(const mrw_skeleton_view *sv, const mrw_clip_view *cA, const mrw_clip_view *cB,
                         float tA, float tB, float w, const float *mask, float *outpal, uint32_t jc){
    static mrw_trs la[RIG_MAX_JOINTS], lb[RIG_MAX_JOINTS], bl[RIG_MAX_JOINTS];
    static float model[RIG_MAX_JOINTS*12];
    mrw_clip_sample_local(cA, tA, la, jc);
    mrw_clip_sample_local(cB, tB, lb, jc);
    mrw_pose_blend(la, lb, w, mask, bl, jc, jc);
    mrw_local_to_model(sv, bl, model, jc);
    mrw_model_to_palette(sv, model, outpal, jc);
}
static void oracle_accum(const mrw_skeleton_view *sv, const mrw_clip_view *cBase, const mrw_trs *delta,
                         float t, float w, const float *mask, float *outpal, uint32_t jc){
    static mrw_trs base[RIG_MAX_JOINTS], acc[RIG_MAX_JOINTS];
    static float model[RIG_MAX_JOINTS*12];
    mrw_clip_sample_local(cBase, t, base, jc);
    mrw_pose_accumulate(base, delta, w, mask, acc, jc, jc);
    mrw_local_to_model(sv, acc, model, jc);
    mrw_model_to_palette(sv, model, outpal, jc);
}

static int cmp_pal(const char *what, const char *bk, int simd, uint32_t jc, uint32_t N,
                   const float *got, const float *ref){
    for (uint32_t i=0;i<N;i++) for (uint32_t k=0;k<jc*12u;k++){
        size_t idx=(size_t)i*jc*12u+k;
        /* scalar batch == looped single-pose reference BIT-FOR-BIT (same ops, same order); SIMD within
         * tolerance (FMA/reassoc). */
        if (!simd) {
            if (got[idx] != ref[idx]) { printf("FAIL %s[%s] (scalar not bit-exact) jc=%u N=%u i=%u j=%u c=%u: %g != %g\n",
                what,bk,jc,N,i,k/12u,k%12u,(double)got[idx],(double)ref[idx]); ++g_fail; return 0; }
            continue;
        }
        double a=got[idx], e=ref[idx], d=fabs(a-e), tol=SIMD_ATOL + SIMD_RTOL*fabs(e);
        if (d>tol){ printf("FAIL %s[%s] jc=%u N=%u i=%u j=%u c=%u: |%g-%g|=%g>%g\n",
            what,bk,jc,N,i,k/12u,k%12u,a,e,d,tol); ++g_fail; return 0; }
    }
    return 1;
}

/* Run blend + accumulate parity for one rig shape across all tail counts. */
static void run(uint32_t jc, uint32_t sc, uint32_t flags, uint32_t seed){
    uint8_t *buf=NULL; size_t sz = build_2clip(jc, sc, flags, seed, &buf);
    mrw_blob blob; CHECK_EQ(mrw_blob_open(buf,sz,&blob), MRW_OK);
    mrw_skeleton_view sv; CHECK_EQ(mrw_blob_skeleton(&blob,&sv), MRW_OK);
    mrw_clip_view cA, cB; CHECK_EQ(mrw_clip_view_at(&blob,1,&cA), MRW_OK); CHECK_EQ(mrw_clip_view_at(&blob,2,&cB), MRW_OK);

    backend_t bks[6]; int nbk = host_backends(bks);
    float dur = (sc>1)?(float)(sc-1)/RIG_FPS:0.0f;
    rig_rng r = { seed*2654435761u + 13u };

    /* a valid shared additive delta = make_additive of two pose arrays (canonical rotation + ratio
     * scale). Keep the delta REALISTIC (small offset, scale ratio near 1): an extreme delta would
     * geometrically explode the deep 65-joint LINEAR chain and amplify ~1-ULP/joint SIMD noise into
     * the output - a pathology of the worst-case rig, not a kernel defect. */
    static mrw_trs pA[RIG_MAX_JOINTS], pB[RIG_MAX_JOINTS], delta[RIG_MAX_JOINTS];
    for (uint32_t j=0;j<jc;j++){
        float q[4]; rig_quat(&r,q); for(int k=0;k<4;k++) pA[j].rot[k]=q[k];
        for(int k=0;k<3;k++){ pA[j].trans[k]=rig_f(&r,-0.3f,0.3f); pA[j].scale[k]=rig_f(&r,0.85f,1.2f); }
        rig_quat(&r,q); for(int k=0;k<4;k++) pB[j].rot[k]=q[k];
        for(int k=0;k<3;k++){ pB[j].trans[k]=rig_f(&r,-0.3f,0.3f); pB[j].scale[k]=rig_f(&r,0.85f,1.2f); }
    }
    CHECK_EQ(mrw_pose_make_additive(pA, pB, delta, jc, jc), MRW_OK);

    float mask[RIG_MAX_JOINTS];
    for (uint32_t j=0;j<jc;j++) mask[j] = (j%4==0)?0.0f:(j%4==1)?1.0f:rig_f(&r,0.0f,1.0f);

    for (size_t ti=0; ti<sizeof TAILS/sizeof TAILS[0]; ++ti){
        uint32_t N = TAILS[ti];
        float *tA=(float*)malloc((size_t)N*4), *tB=(float*)malloc((size_t)N*4), *wts=(float*)malloc((size_t)N*4);
        for (uint32_t i=0;i<N;i++){
            tA[i]=(sc>1)?rig_f(&r,-1.5f*dur,1.5f*dur):rig_f(&r,-5,5);
            tB[i]=(sc>1)?rig_f(&r,-1.5f*dur,1.5f*dur):rig_f(&r,-5,5);
            wts[i]=rig_f(&r,-0.2f,1.2f);   /* includes out-of-range (clamped) weights */
        }
        mrw_mem_req sreq,oreq,oreq16;
        CHECK_EQ(mrw_batch_blend_clips_to_palette_requirements(jc,N,MRW_PALETTE_F32,&sreq,&oreq), MRW_OK);
        CHECK_EQ(mrw_batch_blend_clips_to_palette_requirements(jc,N,MRW_PALETTE_F16,NULL,&oreq16), MRW_OK);
        uint8_t *scr = mrw_alloc64(sreq.size?sreq.size:64);

        /* two mask variants: NULL (uniform) and the per-joint shared mask */
        const float *masks[2] = { NULL, mask };
        for (int mi=0; mi<2; ++mi){
            const float *mk = masks[mi];
            static float refB[RIG_MAX_JOINTS*12*100], refAcc[RIG_MAX_JOINTS*12*100];
            for (uint32_t i=0;i<N;i++){
                oracle_blend(&sv,&cA,&cB,tA[i],tB[i],wts[i],mk, refB+(size_t)i*jc*12u, jc);
                oracle_accum(&sv,&cA,delta,tA[i],wts[i],mk, refAcc+(size_t)i*jc*12u, jc);
            }
            for (int b=0;b<nbk;b++){
                size_t cap=oreq.size+64; uint8_t *ob=mrw_alloc64(cap); memset(ob,0xAB,cap); float *out=(float*)ob;
                /* blend f32 */
                CHECK_EQ(mrw_batch_blend_clips_to_palette(&bks[b].d,&sv,&cA,&cB,tA,tB,wts,mk,N,out,oreq.size,scr,sreq.size), MRW_OK);
                cmp_pal("blend", bks[b].name, bks[b].simd, jc, N, out, refB);
                for (size_t k=oreq.size;k<cap;k++) if (ob[k]!=0xAB){ printf("FAIL blend-canary[%s] jc=%u N=%u\n",bks[b].name,jc,N); ++g_fail; break; }
                /* blend f16: narrowed-f32 consistency + decode parity */
                uint8_t *o16=mrw_alloc64(oreq16.size+64); memset(o16,0xAB,oreq16.size+64); uint16_t *h=(uint16_t*)o16;
                CHECK_EQ(mrw_batch_blend_clips_to_palette_f16(&bks[b].d,&sv,&cA,&cB,tA,tB,wts,mk,N,h,oreq16.size,scr,sreq.size), MRW_OK);
                for (uint32_t i=0;i<N && !g_fail;i++) for (uint32_t k=0;k<jc*12u;k++){
                    size_t idx=(size_t)i*jc*12u+k; uint16_t hx=mrw_f32_to_f16(out[idx]);
                    if (h[idx]!=hx){ printf("FAIL blend-f16-consistency[%s] jc=%u N=%u\n",bks[b].name,jc,N); ++g_fail; break; }
                    double a=mrw_half_to_float(h[idx]), e=refB[idx], d=fabs(a-e);
                    double tol=(bks[b].simd?(SIMD_ATOL+SIMD_RTOL*fabs(e)):0.0)+f16_half_ulp(e);
                    if (d>tol){ printf("FAIL blend-f16[%s] jc=%u N=%u: |%g-%g|=%g>%g\n",bks[b].name,jc,N,a,e,d,tol); ++g_fail; break; }
                }
                mrw_free(o16);
                /* accumulate f32 + f16 */
                memset(ob,0xAB,cap);
                CHECK_EQ(mrw_batch_accumulate_to_palette(&bks[b].d,&sv,&cA,tA,delta,wts,mk,N,out,oreq.size,scr,sreq.size), MRW_OK);
                cmp_pal("accum", bks[b].name, bks[b].simd, jc, N, out, refAcc);
                uint8_t *a16=mrw_alloc64(oreq16.size+64); memset(a16,0xAB,oreq16.size+64); uint16_t *ah=(uint16_t*)a16;
                CHECK_EQ(mrw_batch_accumulate_to_palette_f16(&bks[b].d,&sv,&cA,tA,delta,wts,mk,N,ah,oreq16.size,scr,sreq.size), MRW_OK);
                for (uint32_t i=0;i<N && !g_fail;i++) for (uint32_t k=0;k<jc*12u;k++){
                    size_t idx=(size_t)i*jc*12u+k; uint16_t hx=mrw_f32_to_f16(out[idx]);
                    if (ah[idx]!=hx){ printf("FAIL accum-f16-consistency[%s] jc=%u N=%u\n",bks[b].name,jc,N); ++g_fail; break; }
                    double a=mrw_half_to_float(ah[idx]), e=refAcc[idx], d=fabs(a-e);
                    double tol=(bks[b].simd?(SIMD_ATOL+SIMD_RTOL*fabs(e)):0.0)+f16_half_ulp(e);
                    if (d>tol){ printf("FAIL accum-f16[%s] jc=%u N=%u: |%g-%g|=%g>%g\n",bks[b].name,jc,N,a,e,d,tol); ++g_fail; break; }
                }
                mrw_free(a16); mrw_free(ob);
            }
        }
        free(tA); free(tB); free(wts); mrw_free(scr);
    }
    mrw_free(buf);
}

/* The dispatch / incompatibility / capacity error contract (mirrors test_batch run_edge, abridged). */
static void run_edge(void){
    uint8_t *buf=NULL; size_t sz=build_2clip(8,5,0,123,&buf);
    mrw_blob blob; CHECK_EQ(mrw_blob_open(buf,sz,&blob), MRW_OK);
    mrw_skeleton_view sv; mrw_clip_view cA,cB;
    mrw_blob_skeleton(&blob,&sv); mrw_clip_view_at(&blob,1,&cA); mrw_clip_view_at(&blob,2,&cB);
    uint32_t jc=8;
    mrw_dispatch d; mrw_dispatch_scalar(&d);
    mrw_mem_req sreq,oreq; CHECK_EQ(mrw_batch_blend_clips_to_palette_requirements(jc,4,MRW_PALETTE_F32,&sreq,&oreq), MRW_OK);
    uint8_t *scr=mrw_alloc64(sreq.size); uint8_t *ob=mrw_alloc64(oreq.size); float *out=(float*)ob;
    float t[4]={0,0.05f,0.1f,0.15f}, w[4]={0.2f,0.4f,0.6f,0.8f};
    static mrw_trs delta[8]; for(int j=0;j<8;j++){ delta[j].rot[0]=delta[j].rot[1]=delta[j].rot[2]=0; delta[j].rot[3]=1;
        for(int k=0;k<3;k++){ delta[j].trans[k]=0; delta[j].scale[k]=1; } }

    /* instance_count==0 no-op (data pointers may be NULL) */
    CHECK_EQ(mrw_batch_blend_clips_to_palette(&d,&sv,&cA,&cB,NULL,NULL,NULL,NULL,0,NULL,0,NULL,0), MRW_OK);
    CHECK_EQ(mrw_batch_accumulate_to_palette(&d,&sv,&cA,NULL,NULL,NULL,NULL,0,NULL,0,NULL,0), MRW_OK);
    /* NULL handles ⇒ RANGE */
    CHECK_EQ(mrw_batch_blend_clips_to_palette(NULL,&sv,&cA,&cB,t,t,w,NULL,4,out,oreq.size,scr,sreq.size), MRW_E_RANGE);
    CHECK_EQ(mrw_batch_blend_clips_to_palette(&d,&sv,NULL,&cB,t,t,w,NULL,4,out,oreq.size,scr,sreq.size), MRW_E_RANGE);
    CHECK_EQ(mrw_batch_accumulate_to_palette(&d,&sv,NULL,t,delta,w,NULL,4,out,oreq.size,scr,sreq.size), MRW_E_RANGE);
    /* NULL data with N>0 ⇒ RANGE */
    CHECK_EQ(mrw_batch_blend_clips_to_palette(&d,&sv,&cA,&cB,NULL,t,w,NULL,4,out,oreq.size,scr,sreq.size), MRW_E_RANGE);
    CHECK_EQ(mrw_batch_accumulate_to_palette(&d,&sv,&cA,t,NULL,w,NULL,4,out,oreq.size,scr,sreq.size), MRW_E_RANGE);
    /* bad dispatch ⇒ UNSUPPORTED */
    mrw_dispatch dbad={ MRW_BACKEND_AVX2, 0 };
    CHECK_EQ(mrw_batch_blend_clips_to_palette(&dbad,&sv,&cA,&cB,t,t,w,NULL,4,out,oreq.size,scr,sreq.size), MRW_E_UNSUPPORTED);
    /* capacity */
    CHECK_EQ(mrw_batch_blend_clips_to_palette(&d,&sv,&cA,&cB,t,t,w,NULL,4,out,oreq.size-1,scr,sreq.size), MRW_E_CAPACITY);
    /* non-finite time/weight ⇒ RANGE, no output (canary) */
    memset(ob,0xAB,oreq.size);
    float bt[4]={0,(float)NAN,0.1f,0.15f};
    CHECK_EQ(mrw_batch_blend_clips_to_palette(&d,&sv,&cA,&cB,bt,t,w,NULL,4,out,oreq.size,scr,sreq.size), MRW_E_RANGE);
    for (size_t k=0;k<oreq.size;k++) if (ob[k]!=0xAB){ printf("FAIL: blend wrote on non-finite time\n"); ++g_fail; break; }
    float bw[4]={0.2f,(float)INFINITY,0.6f,0.8f};
    CHECK_EQ(mrw_batch_accumulate_to_palette(&d,&sv,&cA,t,delta,bw,NULL,4,out,oreq.size,scr,sreq.size), MRW_E_RANGE);
    mrw_free(scr); mrw_free(ob); mrw_free(buf);

    /* incompatible: clip from a different skeleton */
    uint8_t *bX=NULL; size_t szX=build_2clip(12,5,0,9,&bX);
    mrw_blob blX; mrw_blob_open(bX,szX,&blX);
    mrw_skeleton_view svX; mrw_clip_view cX; mrw_blob_skeleton(&blX,&svX); mrw_clip_view_at(&blX,1,&cX);
    mrw_mem_req sr2,or2; mrw_batch_blend_clips_to_palette_requirements(8,4,MRW_PALETTE_F32,&sr2,&or2);
    uint8_t *s2=mrw_alloc64(sr2.size), *o2=mrw_alloc64(or2.size);
    CHECK_EQ(mrw_batch_blend_clips_to_palette(&d,&sv,&cX,&cB,t,t,w,NULL,4,(float*)o2,or2.size,s2,sr2.size), MRW_E_INCOMPATIBLE);
    mrw_free(s2); mrw_free(o2); mrw_free(bX);
}

int main(void){
    { backend_t bk[6]; int n=host_backends(bk); printf("backends:"); for(int i=0;i<n;i++) printf(" %s",bk[i].name); printf("\n"); }
    uint32_t bones[] = { 8, 20, 65 };
    for (size_t b=0;b<sizeof bones/sizeof bones[0];++b){
        run(bones[b], 1, 0,                100u+bones[b]);   /* static clips */
        run(bones[b], 6, 0,                200u+bones[b]);   /* non-looping  */
        run(bones[b], 8, MRW_CLIP_LOOPING, 300u+bones[b]);   /* looping      */
    }
    run_edge();
    printf(g_fail ? "test_blend_batch: %d FAILED\n" : "test_blend_batch: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
