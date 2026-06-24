/* f16 palette precision gate:
 * decode the f16 output palette, skin a synthetic mesh with it, and measure the skinned-vertex
 * displacement and normal angular error vs the f32 reference palette. f16 is judged END-TO-END on
 * real skinning, NOT from bone-origin translation - because M = model · inverse_bind and the
 * inverse-bind can carry non-uniform scale/shear (so the basis is not guaranteed an O(1) rotation).
 * This rig deliberately gives ~half its joints non-uniform BIND scale to put scale into inverse_bind,
 * and keeps the hierarchy character-scale (bounded-depth tree, small offsets) so skinned positions
 * stay within a ≤3 m character-local extent.
 *
 * Gate: position ≤ 1 mm mean / ≤ 1 cm p100 at ≤3 m extent; normal ≤ 0.5°.
 * This is the portable, asset-free numerical gate. */
#include "test_util.h"
#include "mrw_build.h"
#include <stdlib.h>
#include <math.h>

#define JC     64u
#define SC      6u
#define NINST  16u
#define NVERT  4000

/* tiny deterministic LCG */
typedef struct { uint32_t s; } rng;
static uint32_t u32(rng *r){ r->s = r->s*1664525u + 1013904223u; return r->s; }
static float    rf(rng *r, float lo, float hi){ return lo + (hi-lo)*((float)(u32(r)>>8)/(float)(1u<<24)); }
static void uquat(rng *r, float q[4]){
    float ax=rf(r,-1,1), ay=rf(r,-1,1), az=rf(r,-1,1), n=sqrtf(ax*ax+ay*ay+az*az);
    if (n<1e-6f){ ax=0; ay=0; az=1; n=1; }
    float a=rf(r,-3.14159265f,3.14159265f), s=sinf(a*0.5f);
    q[0]=ax/n*s; q[1]=ay/n*s; q[2]=az/n*s; q[3]=cosf(a*0.5f);
}

typedef struct { float x,y,z, nx,ny,nz; uint32_t nj; uint32_t j[4]; float w[4]; } Vert;

/* Linear-blend skin one vertex with one instance's palette (jc*12 f32, AoS 3×4). Position via
 * M·[p,1]; normal via the blended 3×3 basis (the f32-vs-f16 delta is what the gate measures). */
static void skin(const float *pal, const Vert *v, double p[3], double nrm[3]) {
    double px=0,py=0,pz=0, b[9]={0};
    for (uint32_t k=0; k<v->nj; ++k) {
        const float *M = pal + (size_t)v->j[k]*12; double w=v->w[k];
        px += w*((double)M[0]*v->x + (double)M[1]*v->y + (double)M[2]*v->z + M[3]);
        py += w*((double)M[4]*v->x + (double)M[5]*v->y + (double)M[6]*v->z + M[7]);
        pz += w*((double)M[8]*v->x + (double)M[9]*v->y + (double)M[10]*v->z + M[11]);
        b[0]+=w*M[0]; b[1]+=w*M[1]; b[2]+=w*M[2];
        b[3]+=w*M[4]; b[4]+=w*M[5]; b[5]+=w*M[6];
        b[6]+=w*M[8]; b[7]+=w*M[9]; b[8]+=w*M[10];
    }
    p[0]=px; p[1]=py; p[2]=pz;
    double n0=b[0]*v->nx+b[1]*v->ny+b[2]*v->nz;
    double n1=b[3]*v->nx+b[4]*v->ny+b[5]*v->nz;
    double n2=b[6]*v->nx+b[7]*v->ny+b[8]*v->nz;
    double ln=sqrt(n0*n0+n1*n1+n2*n2); if (ln<1e-12) ln=1;
    nrm[0]=n0/ln; nrm[1]=n1/ln; nrm[2]=n2/ln;
}

int main(void) {
    /* ---- bounded-depth, character-scale rig with non-uniform bind scale on ~half the joints ---- */
    static uint16_t parent[JC];
    static float rest[JC*10], ib[JC*12], samples[JC*SC*10];
    static char nb[JC][8]; static const char *names[JC];
    rng r = { 12345u };
    for (uint32_t j=0;j<JC;++j) {
        parent[j] = (j==0)?0xFFFFu:(uint16_t)((j-1)/3);   /* ternary tree ⇒ depth ≈ log3(64) ≈ 4 */
        float q[4]; uquat(&r,q);
        rest[j*10+0]=q[0]; rest[j*10+1]=q[1]; rest[j*10+2]=q[2]; rest[j*10+3]=q[3];
        rest[j*10+4]=(j==0)?0.0f:rf(&r,-0.3f,0.3f);       /* small offsets ⇒ joints stay near origin */
        rest[j*10+5]=rf(&r,-0.3f,0.3f);
        rest[j*10+6]=rf(&r,-0.3f,0.3f);
        float sc = (j&1u) ? rf(&r,0.7f,1.4f) : 1.0f;      /* half the joints: non-uniform bind scale */
        rest[j*10+7]=sc; rest[j*10+8]=(j&1u)?rf(&r,0.7f,1.4f):1.0f; rest[j*10+9]=(j&1u)?rf(&r,0.7f,1.4f):1.0f;
        snprintf(nb[j],sizeof nb[j],"j%u",j); names[j]=nb[j];
    }
    compute_bind_inverse(JC, parent, rest, ib);
    for (uint32_t j=0;j<JC;++j) for (uint32_t s=0;s<SC;++s) {
        float *smp = samples + ((size_t)j*SC+s)*10; float q[4]; uquat(&r,q);
        smp[0]=q[0]; smp[1]=q[1]; smp[2]=q[2]; smp[3]=q[3];
        smp[4]=rf(&r,-0.3f,0.3f); smp[5]=rf(&r,-0.3f,0.3f); smp[6]=rf(&r,-0.3f,0.3f);
        smp[7]=smp[8]=smp[9]=1.0f;                        /* unit clip scale (codec-0) */
    }
    mrw_skel skel = { JC, parent, rest, ib, names };
    mrw_clip clipd = { 30.0f, SC, 0, samples, NULL };
    uint8_t *buf=NULL; size_t sz = mrw_build(&skel,&clipd,1,NULL,&buf);
    mrw_blob blob; CHECK_EQ(mrw_blob_open(buf,sz,&blob),MRW_OK);
    mrw_skeleton_view sv; CHECK_EQ(mrw_blob_skeleton(&blob,&sv),MRW_OK);
    mrw_clip_view cv; CHECK_EQ(mrw_clip_view_at(&blob,1,&cv),MRW_OK);

    /* ---- synthetic skinned mesh: vertices near the rig, 1–4 joint influences, unit normals ---- */
    static Vert verts[NVERT];
    for (int i=0;i<NVERT;++i) {
        Vert *v=&verts[i];
        v->x=rf(&r,-0.6f,0.6f); v->y=rf(&r,-0.6f,0.6f); v->z=rf(&r,-0.6f,0.6f);
        float nx=rf(&r,-1,1),ny=rf(&r,-1,1),nz=rf(&r,-1,1),nl=sqrtf(nx*nx+ny*ny+nz*nz); if(nl<1e-6f){nx=0;ny=0;nz=1;nl=1;}
        v->nx=nx/nl; v->ny=ny/nl; v->nz=nz/nl;
        v->nj=1u+(u32(&r)&3u); float wsum=0;
        for (uint32_t k=0;k<v->nj;++k){ v->j[k]=u32(&r)%JC; v->w[k]=rf(&r,0.1f,1.0f); wsum+=v->w[k]; }
        for (uint32_t k=0;k<v->nj;++k) v->w[k]/=wsum;
    }

    /* ---- f32 + f16 palettes for NINST instances (scalar backend ⇒ pure f16-quantization error) ---- */
    mrw_dispatch d; mrw_dispatch_scalar(&d);
    mrw_mem_req sreq, oreq, oreq16;
    CHECK_EQ(mrw_batch_clip_to_palette_requirements(JC,NINST,MRW_PALETTE_F32,&sreq,&oreq),MRW_OK);
    CHECK_EQ(mrw_batch_clip_to_palette_requirements(JC,NINST,MRW_PALETTE_F16,NULL,&oreq16),MRW_OK);
    uint8_t *scr=mrw_alloc64(sreq.size);
    float *pal=(float*)mrw_alloc64(oreq.size);
    uint16_t *pal16=(uint16_t*)mrw_alloc64(oreq16.size);
    float times[NINST];
    for (uint32_t i=0;i<NINST;++i) times[i]=rf(&r,0.0f,(float)(SC-1)/30.0f);
    CHECK_EQ(mrw_batch_clip_to_palette(&d,&sv,&cv,times,NINST,pal,oreq.size,scr,sreq.size),MRW_OK);
    CHECK_EQ(mrw_batch_clip_to_palette_f16(&d,&sv,&cv,times,NINST,pal16,oreq16.size,scr,sreq.size),MRW_OK);

    /* ---- skin with both, measure displacement (mm) and normal angle (deg) ---- */
    static float deci[JC*12];   /* decoded f16 palette for one instance */
    double pos_sum=0, pos_max=0, nrm_max=0, ext_max=0; long count=0;
    for (uint32_t i=0;i<NINST;++i) {
        const float *p32 = pal + (size_t)i*JC*12;
        const uint16_t *p16 = pal16 + (size_t)i*JC*12;
        for (uint32_t c=0;c<JC*12u;++c) deci[c]=mrw_half_to_float(p16[c]);
        for (int vi=0;vi<NVERT;++vi) {
            double a[3],an[3], b[3],bn[3];
            skin(p32, &verts[vi], a, an);
            skin(deci, &verts[vi], b, bn);
            double dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
            double dp=sqrt(dx*dx+dy*dy+dz*dz);
            pos_sum+=dp; if(dp>pos_max)pos_max=dp; ++count;
            double ext=sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); if(ext>ext_max)ext_max=ext;
            double dot=an[0]*bn[0]+an[1]*bn[1]+an[2]*bn[2]; if(dot>1)dot=1; if(dot<-1)dot=-1;
            double ang=acos(dot)*180.0/M_PI; if(ang>nrm_max)nrm_max=ang;
        }
    }
    double pos_mean=pos_sum/(double)count;
    printf("f16 skin gate: extent=%.2f m  pos mean=%.4f mm  pos p100=%.4f mm  normal max=%.4f deg\n",
           ext_max, pos_mean*1000.0, pos_max*1000.0, nrm_max);

    /* Tolerances: the extent is character-local (≤3 m), and f16 holds skinning error
     * comfortably under the bar even with scale/shear in inverse_bind. */
    CHECK(ext_max <= 3.0);
    CHECK(pos_mean <= 1e-3);   /* ≤ 1 mm mean */
    CHECK(pos_max  <= 1e-2);   /* ≤ 1 cm p100 */
    CHECK(nrm_max  <= 0.5);    /* ≤ 0.5°      */

    mrw_free(scr); mrw_free((uint8_t*)pal); mrw_free((uint8_t*)pal16); mrw_free(buf);
    printf(g_fail ? "test_f16_skin: %d FAILED\n" : "test_f16_skin: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
