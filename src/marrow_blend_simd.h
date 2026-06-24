/* marrow pose-combine SIMD kernels - across-instance blend / accumulate to palette.
 *
 * ONE source. Each ISA translation unit (marrow_blend_{sse2,avx2,avx2_fma}.c) defines its MRW_ISA_*
 * selector and the kernel names MRW_BLEND_KERNEL / MRW_ACCUM_KERNEL, then includes this file. The
 * per-joint pipeline reuses the shared wide helpers (marrow_batch_kernel.h): the SAME keyframe
 * gather, nlerp/lerp interp, TRS→affine, affine compose, and SoA→AoS scatter as the clip kernel -
 * only the per-instance COMBINE (a 2-clip blend, or a shared-delta accumulate) is new, and it is
 * nlerp/lerp/quat-mul (no trig, no pow), so it vectorizes in the same `wide` vocabulary.
 *
 * Inputs are pre-validated by the dispatcher in marrow_blend_batch.c; `model` is the validated SoA
 * scratch. Result is bit-for-bit-close to the scalar reference (FMA fusion / reassociation only). */
#include "marrow_batch_kernel.h"

#if !defined(MRW_BLEND_KERNEL) || !defined(MRW_ACCUM_KERNEL)
#  error "define MRW_BLEND_KERNEL and MRW_ACCUM_KERNEL before including marrow_blend_simd.h"
#endif

/* Per-clip per-tile sampling context (the gather seam differs by ISA, like the clip kernel). */
typedef struct {
    const mrw_clip_view *clip;
    int st; uint32_t codec, sample_count;
    const uint32_t *i0; const float *uu;
#if defined(MRW_ISA_AVX2) && !defined(MRW_AVX2_GATHER_SCALAR)
    const mrw_blob_f32 *sbase; uint32_t scw;   /* the per-lane vgather index is rebuilt at use */
#else
    const uint8_t *sbytes;
#endif
} mrw_samp;

/* per-lane frame state for one clip over a tile; padded lanes [live,LANES) take times[base]. */
static inline void mrw_fill_frames(const mrw_clip_view *clip, const float *times,
                                   uint32_t base, uint32_t live, uint32_t *i0, float *uu, int *st) {
    int s = 0;
    for (uint32_t l = 0; l < MRW_LANES; ++l) {
        float t = times[base + (l < live ? l : 0)];
        uint32_t fi = 0; float fu = 0.0f;
        s = mrw_clip_frame(clip, t, &fi, &fu);
        i0[l] = fi; uu[l] = fu;
    }
    *st = s;
}

/* Per-tile per-clip AVX2 gather index i0[l]·scw (8 lanes; SUB==1) - hoisted out of the joint loop. */
#if defined(MRW_ISA_AVX2) && !defined(MRW_AVX2_GATHER_SCALAR)
static inline __m256i mrw_samp_gidx(const mrw_samp *cx) {
    return _mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(const void *)cx->i0),
                              _mm256_set1_epi32((int)cx->scw));
}
#  define MRW_GIDX_PARAM , __m256i gidx
#  define MRW_GIDX_ARG(g) , (g)
#else
#  define MRW_GIDX_PARAM
#  define MRW_GIDX_ARG(g)
#endif

/* Sample one clip's joint j into a `wide` TRS (quat xyzw, trans xyz, scale xyz) for subgroup lane L.
 * Static clip ⇒ broadcast sample 0; else gather the two bracket keyframes (per ISA) + interp.
 * `gidx` (AVX2 vgather only) is the per-tile lane index computed once by mrw_samp_gidx. */
static inline void mrw_blend_sample(const mrw_samp *cx MRW_GIDX_PARAM, uint32_t j, uint32_t L,
                                    mrw_w *qx, mrw_w *qy, mrw_w *qz, mrw_w *qw,
                                    mrw_w t[3], mrw_w s[3]) {
    if (cx->st) {
        mrw_trs s0; mrw_clip_sample(cx->clip, j, 0, &s0);
        *qx = mrw_w_set1(s0.rot[0]); *qy = mrw_w_set1(s0.rot[1]);
        *qz = mrw_w_set1(s0.rot[2]); *qw = mrw_w_set1(s0.rot[3]);
        t[0] = mrw_w_set1(s0.trans[0]); t[1] = mrw_w_set1(s0.trans[1]); t[2] = mrw_w_set1(s0.trans[2]);
        s[0] = mrw_w_set1(s0.scale[0]); s[1] = mrw_w_set1(s0.scale[1]); s[2] = mrw_w_set1(s0.scale[2]);
        return;
    }
    mrw_w A_[10], B_[10];
#if defined(MRW_ISA_AVX2) && !defined(MRW_AVX2_GATHER_SCALAR)
    if (cx->codec) mrw_gather_vg_c1(cx->sbase + (size_t)j * cx->sample_count * cx->scw, gidx, A_, B_);
    else           mrw_gather_vg   (cx->sbase + (size_t)j * cx->sample_count * cx->scw, gidx, A_, B_);
#else
    if (cx->codec) mrw_gather_sc_c1(cx->sbytes + (size_t)j * cx->sample_count * MRW_ROOT_SAMPLE_STRIDE, cx->i0, L, A_, B_);
    else           mrw_gather_sc   (cx->sbytes + (size_t)j * cx->sample_count * MRW_CLIP_SAMPLE_STRIDE, cx->i0, L, A_, B_);
#endif
    mrw_w U = mrw_w_loadu(&cx->uu[L]);
    mrw_quat_nlerp_w(A_[0], A_[1], A_[2], A_[3], B_[0], B_[1], B_[2], B_[3], U, qx, qy, qz, qw);
    t[0] = mrw_w_fma(U, mrw_w_sub(B_[4], A_[4]), A_[4]);
    t[1] = mrw_w_fma(U, mrw_w_sub(B_[5], A_[5]), A_[5]);
    t[2] = mrw_w_fma(U, mrw_w_sub(B_[6], A_[6]), A_[6]);
    s[0] = mrw_w_fma(U, mrw_w_sub(B_[7], A_[7]), A_[7]);
    s[1] = mrw_w_fma(U, mrw_w_sub(B_[8], A_[8]), A_[8]);
    s[2] = mrw_w_fma(U, mrw_w_sub(B_[9], A_[9]), A_[9]);
}

static inline void mrw_samp_init_const(mrw_samp *cx, const mrw_clip_view *clip) {
    cx->clip = clip; cx->codec = clip->codec; cx->sample_count = clip->sample_count;
#if defined(MRW_ISA_AVX2) && !defined(MRW_AVX2_GATHER_SCALAR)
    cx->sbase = (const mrw_blob_f32 *)(const void *)(clip->base + clip->samples_off);
    cx->scw = cx->codec ? 7u : 10u;
#else
    cx->sbytes = clip->base + clip->samples_off;
#endif
}

/* compose model(j) = (root ? local : parent_model ∘ local), fold M_j = model(j) ∘ inverse_bind,
 * and scatter - identical to the clip kernel. `la` is the combined local affine for this subgroup. */
static inline void mrw_compose_scatter(const mrw_w la[12], float *model, const mrw_w ibw[12],
                                       int is_root, uint16_t parent, uint32_t L,
                                       uint32_t j, uint32_t jc, uint32_t base, uint32_t live,
                                       void *out_palettes, mrw_palette_format out_format, int use_f16c
#if !defined(MRW_ISA_AVX2)
                                       , float *pal_soa
#endif
                                       ) {
    mrw_w mj[12];
    if (is_root) { for (int c = 0; c < 12; ++c) mj[c] = la[c]; }
    else {
        mrw_w pm[12];
        for (int c = 0; c < 12; ++c) pm[c] = mrw_w_load(&MRW_MODEL_AT(model, parent, c, L));
        mrw_affine_mul_w(pm, la, mj);
    }
    for (int c = 0; c < 12; ++c) mrw_w_store(&MRW_MODEL_AT(model, j, c, L), mj[c]);
    mrw_w Mj[12]; mrw_affine_mul_w(mj, ibw, Mj);
#if defined(MRW_ISA_AVX2)
    mrw_scatter_palette(Mj, out_palettes, out_format, use_f16c, base, jc, j, live);
#else
    (void)jc; (void)base; (void)live; (void)out_palettes; (void)out_format; (void)use_f16c;
    for (int c = 0; c < 12; ++c) mrw_w_store(&pal_soa[(size_t)c * MRW_LANES + L], Mj[c]);
#endif
}

/* SSE2 SoA→AoS deinterleave (live lanes only); no-op on AVX2 (scatter already wrote the AoS). */
static inline void mrw_drain_soa(const float *pal_soa, void *out_palettes, mrw_palette_format out_format,
                                 uint32_t j, uint32_t jc, uint32_t base, uint32_t live) {
#if !defined(MRW_ISA_AVX2)
    for (uint32_t l = 0; l < live; ++l) {
        size_t off = ((size_t)(base + l) * jc + j) * MRW_PALETTE_FLOATS;
        if (out_format == MRW_PALETTE_F16) {
            uint16_t *dst = (uint16_t *)out_palettes + off;
            for (int c = 0; c < 12; ++c) dst[c] = mrw_f32_to_f16(pal_soa[(size_t)c * MRW_LANES + l]);
        } else {
            float *dst = (float *)out_palettes + off;
            for (int c = 0; c < 12; ++c) dst[c] = pal_soa[(size_t)c * MRW_LANES + l];
        }
    }
#else
    (void)pal_soa; (void)out_palettes; (void)out_format; (void)j; (void)jc; (void)base; (void)live;
#endif
}

/* ------------------------------------------------------------------ blend kernel */

MRW_DECL_BLEND_KERNEL(MRW_BLEND_KERNEL) {
    const uint32_t jc  = skel->joint_count;
    const uint32_t SUB = MRW_LANES / MRW_W;
    mrw_samp cxA, cxB; mrw_samp_init_const(&cxA, clipA); mrw_samp_init_const(&cxB, clipB);
    uint32_t i0A[MRW_LANES], i0B[MRW_LANES]; float uuA[MRW_LANES], uuB[MRW_LANES], wbuf[MRW_LANES];
    cxA.i0 = i0A; cxA.uu = uuA; cxB.i0 = i0B; cxB.uu = uuB;

    for (uint32_t base = 0; base < instance_count; ) {
        uint32_t live = instance_count - base; if (live > MRW_LANES) live = MRW_LANES;
        mrw_fill_frames(clipA, timesA, base, live, i0A, uuA, &cxA.st);
        mrw_fill_frames(clipB, timesB, base, live, i0B, uuB, &cxB.st);
        for (uint32_t l = 0; l < MRW_LANES; ++l) wbuf[l] = weights[base + (l < live ? l : 0)];
#if defined(MRW_ISA_AVX2) && !defined(MRW_AVX2_GATHER_SCALAR)
        __m256i gA = mrw_samp_gidx(&cxA), gB = mrw_samp_gidx(&cxB);   /* per-tile, hoisted out of j */
#endif
        for (uint32_t j = 0; j < jc; ++j) {
            int is_root = (j == 0);
            uint16_t parent = is_root ? 0
                : mrw_rd_u16(skel->base + skel->parent_off + (size_t)j * MRW_PARENT_STRIDE);
            float ib[12];
            memcpy(ib, skel->base + skel->inverse_bind_off + (size_t)j * MRW_INVERSE_BIND_STRIDE,
                   MRW_INVERSE_BIND_STRIDE);
            mrw_w ibw[12]; for (int c = 0; c < 12; ++c) ibw[c] = mrw_w_set1(ib[c]);
            mrw_w maskj = mask ? mrw_w_set1(mrw_clampf(mask[j], 0.0f, 1.0f)) : mrw_w_set1(1.0f);
#if !defined(MRW_ISA_AVX2)
            _Alignas(MRW_W_ALIGN) float pal_soa[MRW_PALETTE_FLOATS * MRW_LANES];
#endif
            for (uint32_t g = 0; g < SUB; ++g) {
                uint32_t L = g * MRW_W;
                mrw_w qAx,qAy,qAz,qAw, tA[3], sA[3];
                mrw_w qBx,qBy,qBz,qBw, tB[3], sB[3];
                mrw_blend_sample(&cxA MRW_GIDX_ARG(gA), j, L, &qAx,&qAy,&qAz,&qAw, tA, sA);
                mrw_blend_sample(&cxB MRW_GIDX_ARG(gB), j, L, &qBx,&qBy,&qBz,&qBw, tB, sB);
                /* we = clamp(clamp(w,0,1)·clamp(mask,0,1), 0,1) */
                mrw_w we = mrw_w_clamp01(mrw_w_mul(mrw_w_clamp01(mrw_w_loadu(&wbuf[L])), maskj));
                mrw_w qx,qy,qz,qw;
                mrw_quat_nlerp_w(qAx,qAy,qAz,qAw, qBx,qBy,qBz,qBw, we, &qx,&qy,&qz,&qw);
                mrw_w tx = mrw_w_fma(we, mrw_w_sub(tB[0],tA[0]), tA[0]);
                mrw_w ty = mrw_w_fma(we, mrw_w_sub(tB[1],tA[1]), tA[1]);
                mrw_w tz = mrw_w_fma(we, mrw_w_sub(tB[2],tA[2]), tA[2]);
                mrw_w sx = mrw_w_fma(we, mrw_w_sub(sB[0],sA[0]), sA[0]);
                mrw_w sy = mrw_w_fma(we, mrw_w_sub(sB[1],sA[1]), sA[1]);
                mrw_w sz = mrw_w_fma(we, mrw_w_sub(sB[2],sA[2]), sA[2]);
                mrw_w la[12]; mrw_trs_to_affine_w(qx,qy,qz,qw, tx,ty,tz, sx,sy,sz, la);
                mrw_compose_scatter(la, model, ibw, is_root, parent, L, j, jc, base, live,
                                    out_palettes, out_format, use_f16c
#if !defined(MRW_ISA_AVX2)
                                    , pal_soa
#endif
                                    );
            }
            mrw_drain_soa(
#if !defined(MRW_ISA_AVX2)
                pal_soa,
#else
                (const float *)0,
#endif
                out_palettes, out_format, j, jc, base, live);
        }
        base += live;
    }
    return MRW_OK;
}

/* ------------------------------------------------------------------ accumulate kernel */

MRW_DECL_ACCUM_KERNEL(MRW_ACCUM_KERNEL) {
    const uint32_t jc  = skel->joint_count;
    const uint32_t SUB = MRW_LANES / MRW_W;
    mrw_samp cx; mrw_samp_init_const(&cx, base_clip);
    uint32_t i0[MRW_LANES]; float uu[MRW_LANES], wbuf[MRW_LANES];
    cx.i0 = i0; cx.uu = uu;
    const mrw_w qiw = mrw_w_set1(1.0f), z = mrw_w_zero();

    for (uint32_t base = 0; base < instance_count; ) {
        uint32_t live = instance_count - base; if (live > MRW_LANES) live = MRW_LANES;
        mrw_fill_frames(base_clip, times, base, live, i0, uu, &cx.st);
        for (uint32_t l = 0; l < MRW_LANES; ++l) wbuf[l] = weights[base + (l < live ? l : 0)];
#if defined(MRW_ISA_AVX2) && !defined(MRW_AVX2_GATHER_SCALAR)
        __m256i gidx = mrw_samp_gidx(&cx);   /* per-tile, hoisted out of j */
#endif
        for (uint32_t j = 0; j < jc; ++j) {
            int is_root = (j == 0);
            uint16_t parent = is_root ? 0
                : mrw_rd_u16(skel->base + skel->parent_off + (size_t)j * MRW_PARENT_STRIDE);
            float ib[12];
            memcpy(ib, skel->base + skel->inverse_bind_off + (size_t)j * MRW_INVERSE_BIND_STRIDE,
                   MRW_INVERSE_BIND_STRIDE);
            mrw_w ibw[12]; for (int c = 0; c < 12; ++c) ibw[c] = mrw_w_set1(ib[c]);
            mrw_w maskj = mask ? mrw_w_set1(mrw_clampf(mask[j], 0.0f, 1.0f)) : mrw_w_set1(1.0f);
            /* shared delta (same for every lane) broadcast once per joint */
            mrw_w dqx=mrw_w_set1(delta[j].rot[0]), dqy=mrw_w_set1(delta[j].rot[1]),
                  dqz=mrw_w_set1(delta[j].rot[2]), dqw=mrw_w_set1(delta[j].rot[3]);
            mrw_w dtx=mrw_w_set1(delta[j].trans[0]), dty=mrw_w_set1(delta[j].trans[1]), dtz=mrw_w_set1(delta[j].trans[2]);
            mrw_w dsx=mrw_w_set1(delta[j].scale[0]), dsy=mrw_w_set1(delta[j].scale[1]), dsz=mrw_w_set1(delta[j].scale[2]);
#if !defined(MRW_ISA_AVX2)
            _Alignas(MRW_W_ALIGN) float pal_soa[MRW_PALETTE_FLOATS * MRW_LANES];
#endif
            for (uint32_t g = 0; g < SUB; ++g) {
                uint32_t L = g * MRW_W;
                mrw_w qbx,qby,qbz,qbw, tb[3], sb[3];
                mrw_blend_sample(&cx MRW_GIDX_ARG(gidx), j, L, &qbx,&qby,&qbz,&qbw, tb, sb);
                mrw_w we = mrw_w_clamp01(mrw_w_mul(mrw_w_clamp01(mrw_w_loadu(&wbuf[L])), maskj));
                /* nlerp_id(delta.rot, we) = nlerp(identity, delta.rot, we); rot = normalize(base ⊗ nid) */
                mrw_w nx,ny,nz,nw;
                mrw_quat_nlerp_w(z,z,z,qiw, dqx,dqy,dqz,dqw, we, &nx,&ny,&nz,&nw);
                mrw_w rx,ry,rz,rw;
                mrw_quat_mul_w(qbx,qby,qbz,qbw, nx,ny,nz,nw, &rx,&ry,&rz,&rw);
                mrw_quat_normalize_w(&rx,&ry,&rz,&rw);
                /* trans = base + we·delta.trans; scale = base + we·(base·delta.scale − base) */
                mrw_w tx = mrw_w_fma(we, dtx, tb[0]);
                mrw_w ty = mrw_w_fma(we, dty, tb[1]);
                mrw_w tz = mrw_w_fma(we, dtz, tb[2]);
                mrw_w sx = mrw_w_fma(we, mrw_w_sub(mrw_w_mul(sb[0],dsx), sb[0]), sb[0]);
                mrw_w sy = mrw_w_fma(we, mrw_w_sub(mrw_w_mul(sb[1],dsy), sb[1]), sb[1]);
                mrw_w sz = mrw_w_fma(we, mrw_w_sub(mrw_w_mul(sb[2],dsz), sb[2]), sb[2]);
                mrw_w la[12]; mrw_trs_to_affine_w(rx,ry,rz,rw, tx,ty,tz, sx,sy,sz, la);
                mrw_compose_scatter(la, model, ibw, is_root, parent, L, j, jc, base, live,
                                    out_palettes, out_format, use_f16c
#if !defined(MRW_ISA_AVX2)
                                    , pal_soa
#endif
                                    );
            }
            mrw_drain_soa(
#if !defined(MRW_ISA_AVX2)
                pal_soa,
#else
                (const float *)0,
#endif
                out_palettes, out_format, j, jc, base, live);
        }
        base += live;
    }
    return MRW_OK;
}
