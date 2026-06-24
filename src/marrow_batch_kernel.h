/* marrow batch wide-kernel HELPERS - across-instance SoA vertical `wide` math.
 *
 * One source for the per-joint pose pipeline (nlerp/lerp, TRS→affine, affine compose) and the
 * per-ISA lane-crossing seams (keyframe gather, SoA→AoS scatter), shared by the clip kernel
 * (marrow_batch_simd.h) and the pose-combine kernels (marrow_blend_simd.h). Each ISA
 * translation unit defines exactly one of MRW_ISA_SSE2 / MRW_ISA_AVX2 (and, for AVX2, optionally
 * MRW_ISA_FMA) before including this header; every wrapper is `static inline`.
 *
 * The vertical math is purely component-wise (no cross-128-lane ops); only the gather and scatter
 * are lane-crossing and are specialized per ISA below. This header carries NO kernel entry point -
 * the including TU defines its own after including this. */
#ifndef MRW_BATCH_KERNEL_H
#define MRW_BATCH_KERNEL_H
#include "marrow_internal.h"
#include "marrow_wide.h"
/* C = A ∘ B over `wide` lanes (apply B then A); mirrors scalar mrw_affine_mul. out aliases none. */
static inline void mrw_affine_mul_w(const mrw_w a[12], const mrw_w b[12], mrw_w out[12]) {
    for (int r = 0; r < 3; ++r) {
        mrw_w a0 = a[4*r+0], a1 = a[4*r+1], a2 = a[4*r+2], at = a[4*r+3];
        out[4*r+0] = mrw_w_fma(a0, b[0], mrw_w_fma(a1, b[4], mrw_w_mul(a2, b[8])));
        out[4*r+1] = mrw_w_fma(a0, b[1], mrw_w_fma(a1, b[5], mrw_w_mul(a2, b[9])));
        out[4*r+2] = mrw_w_fma(a0, b[2], mrw_w_fma(a1, b[6], mrw_w_mul(a2, b[10])));
        out[4*r+3] = mrw_w_fma(a0, b[3], mrw_w_fma(a1, b[7], mrw_w_fma(a2, b[11], at)));
    }
}

/* TRS (quat x,y,z,w; trans; non-uniform scale) → 12-component 3×4 affine, per `wide` lane.
 * Mirrors mrw_quat_to_mat3 + compose_affine. */
static inline void mrw_trs_to_affine_w(mrw_w x, mrw_w y, mrw_w z, mrw_w w,
                                       mrw_w tx, mrw_w ty, mrw_w tz,
                                       mrw_w sx, mrw_w sy, mrw_w sz, mrw_w out[12]) {
    mrw_w two = mrw_w_set1(2.0f), one = mrw_w_set1(1.0f);
    mrw_w xx = mrw_w_mul(x,x), yy = mrw_w_mul(y,y), zz = mrw_w_mul(z,z);
    mrw_w xy = mrw_w_mul(x,y), xz = mrw_w_mul(x,z), yz = mrw_w_mul(y,z);
    mrw_w wx = mrw_w_mul(w,x), wy = mrw_w_mul(w,y), wz = mrw_w_mul(w,z);
    mrw_w r0 = mrw_w_sub(one, mrw_w_mul(two, mrw_w_add(yy, zz)));
    mrw_w r1 = mrw_w_mul(two, mrw_w_sub(xy, wz));
    mrw_w r2 = mrw_w_mul(two, mrw_w_add(xz, wy));
    mrw_w r3 = mrw_w_mul(two, mrw_w_add(xy, wz));
    mrw_w r4 = mrw_w_sub(one, mrw_w_mul(two, mrw_w_add(xx, zz)));
    mrw_w r5 = mrw_w_mul(two, mrw_w_sub(yz, wx));
    mrw_w r6 = mrw_w_mul(two, mrw_w_sub(xz, wy));
    mrw_w r7 = mrw_w_mul(two, mrw_w_add(yz, wx));
    mrw_w r8 = mrw_w_sub(one, mrw_w_mul(two, mrw_w_add(xx, yy)));
    out[0]=mrw_w_mul(r0,sx); out[1]=mrw_w_mul(r1,sy); out[2] =mrw_w_mul(r2,sz); out[3] =tx;
    out[4]=mrw_w_mul(r3,sx); out[5]=mrw_w_mul(r4,sy); out[6] =mrw_w_mul(r5,sz); out[7] =ty;
    out[8]=mrw_w_mul(r6,sx); out[9]=mrw_w_mul(r7,sy); out[10]=mrw_w_mul(r8,sz); out[11]=tz;
}

/* Scale-free variant for codec-1 clips (scale ≡ (1,1,1)): identical to mrw_trs_to_affine_w with
 * sx=sy=sz=1, but drops the 9 column·scale multiplies. Bit-identical to multiplying by exactly 1.0f
 * (an exact op), so it agrees with the scalar reference (which samples scale=1 and multiplies). */
static inline void mrw_trs_to_affine_w_ns(mrw_w x, mrw_w y, mrw_w z, mrw_w w,
                                          mrw_w tx, mrw_w ty, mrw_w tz, mrw_w out[12]) {
    mrw_w two = mrw_w_set1(2.0f), one = mrw_w_set1(1.0f);
    mrw_w xx = mrw_w_mul(x,x), yy = mrw_w_mul(y,y), zz = mrw_w_mul(z,z);
    mrw_w xy = mrw_w_mul(x,y), xz = mrw_w_mul(x,z), yz = mrw_w_mul(y,z);
    mrw_w wx = mrw_w_mul(w,x), wy = mrw_w_mul(w,y), wz = mrw_w_mul(w,z);
    out[0]=mrw_w_sub(one, mrw_w_mul(two, mrw_w_add(yy, zz))); out[1]=mrw_w_mul(two, mrw_w_sub(xy, wz)); out[2] =mrw_w_mul(two, mrw_w_add(xz, wy)); out[3] =tx;
    out[4]=mrw_w_mul(two, mrw_w_add(xy, wz)); out[5]=mrw_w_sub(one, mrw_w_mul(two, mrw_w_add(xx, zz))); out[6] =mrw_w_mul(two, mrw_w_sub(yz, wx)); out[7] =ty;
    out[8]=mrw_w_mul(two, mrw_w_sub(xz, wy)); out[9]=mrw_w_mul(two, mrw_w_add(yz, wx)); out[10]=mrw_w_sub(one, mrw_w_mul(two, mrw_w_add(xx, yy))); out[11]=tz;
}

/* hemisphere-corrected nlerp of two quats per `wide` lane (mirrors mrw_quat_nlerp).
 * b is negated iff dot < 0 (branchless), then normalize with sqrt+div (full precision - matches
 * scalar 1/sqrtf; no rsqrt approximation). The flip uses a `dot < 0` compare, NOT the raw sign
 * bit of dot: the scalar reference flips on `dot < 0.0f`, which is FALSE for dot == -0.0, but a raw
 * sign-bit test would flip on -0.0 (a value the loader accepts) and pick the opposite hemisphere. */
static inline void mrw_quat_nlerp_w(mrw_w ax, mrw_w ay, mrw_w az, mrw_w aw,
                                    mrw_w bx, mrw_w by, mrw_w bz, mrw_w bw, mrw_w u,
                                    mrw_w *ox, mrw_w *oy, mrw_w *oz, mrw_w *ow) {
    mrw_w dot  = mrw_w_fma(ax,bx, mrw_w_fma(ay,by, mrw_w_fma(az,bz, mrw_w_mul(aw,bw))));
    mrw_w sign = mrw_w_and(mrw_w_cmpgt(mrw_w_zero(), dot), mrw_w_set1(-0.0f)); /* dot<0 → -0.0 sign */
    mrw_w sbx = mrw_w_xor(bx, sign), sby = mrw_w_xor(by, sign);     /* s·b */
    mrw_w sbz = mrw_w_xor(bz, sign), sbw = mrw_w_xor(bw, sign);
    mrw_w qx = mrw_w_fma(u, mrw_w_sub(sbx, ax), ax);               /* a + u·(s·b − a) */
    mrw_w qy = mrw_w_fma(u, mrw_w_sub(sby, ay), ay);
    mrw_w qz = mrw_w_fma(u, mrw_w_sub(sbz, az), az);
    mrw_w qw = mrw_w_fma(u, mrw_w_sub(sbw, aw), aw);
    mrw_w n2  = mrw_w_fma(qx,qx, mrw_w_fma(qy,qy, mrw_w_fma(qz,qz, mrw_w_mul(qw,qw))));
    mrw_w inv = mrw_w_div(mrw_w_set1(1.0f), mrw_w_sqrt(n2));
    inv = mrw_w_select(mrw_w_cmpgt(n2, mrw_w_zero()), inv, mrw_w_zero()); /* n2≤0 → 0 (scalar guard) */
    *ox = mrw_w_mul(qx, inv); *oy = mrw_w_mul(qy, inv);
    *oz = mrw_w_mul(qz, inv); *ow = mrw_w_mul(qw, inv);
}

/* ---- per-ISA lane-crossing seam: keyframe gather ----
 * The vertical pose math above is ONE source for both ISAs; the lane-crossing keyframe gather is
 * not. Each instance (lane) reads a different frame i0[lane], so this is a genuine per-lane gather
 * off the `wide` path. AVX2 has a hardware gather, so it loads the 8 lanes' samples straight from
 * the validated blob in 8-wide vmovgather steps; SSE2 (no gather) inlines the 40-byte sample read
 * (no call into the external mrw_clip_sample) and packs per lane. Both fill A_/B_ = the two bracket
 * keyframes as 10 `wide` components (rot xyzw, trans xyz, scale xyz), which the shared nlerp/lerp
 * then consume. Folding the gather into the kernel (vs. routing every lane through mrw_clip_sample)
 * is what removes the ISA-independent per-lane cost that was throttling AVX2 to ~SSE2 speed.
 *
 * AVX2 defaults to hardware vgather - measured the fastest here (Zen4: ~20% over the manual
 * deinterleave; on slow-gather µarches like Zen2 it can lose, so even AVX2 may prefer the manual
 * path there). Define MRW_AVX2_GATHER_SCALAR to build the AVX2 TUs with the SSE2-style scalar
 * deinterleave instead - same parity, a per-target build knob, not a runtime branch. */
#if defined(MRW_ISA_AVX2)
#if !defined(MRW_AVX2_GATHER_SCALAR)
/* may_alias float view of the immutable .mrw blob. Hardware gather needs a typed base, so this is
 * the one spot marrow reads blob floats without the loader's memcpy.
 * It is well-defined under -fstrict-aliasing because the blob is READ-ONLY and every other access to
 * it is byte-wise (memcpy / mrw_rd_*, i.e. char - which aliases all types): there is no differently-
 * typed store anywhere for the optimizer to reorder this read against. may_alias states that intent
 * to GCC/Clang; MSVC does no type-based alias analysis here and needs no annotation. */
#if defined(__GNUC__) || defined(__clang__)
typedef float __attribute__((__may_alias__)) mrw_blob_f32;
#else
typedef float mrw_blob_f32;
#endif
/* AVX2 hardware gather: A=sample(i0), B=sample(i0+1) for 10 components × 8 lanes. gidx10[l]=i0[l]*10
 * is the float-unit index of lane l's frame within joint j's sample run; jbase points at joint j's
 * first sample. Clip samples are validated binary32; the index folds j into jbase so it never
 * leaves this joint's small sample range. */
static inline void mrw_gather_vg(const mrw_blob_f32 *jbase, __m256i gidx10, mrw_w A_[10], mrw_w B_[10]) {
    for (int c = 0; c < 10; ++c) {
        A_[c] = _mm256_i32gather_ps((const float *)jbase, _mm256_add_epi32(gidx10, _mm256_set1_epi32(c)),      4);
        B_[c] = _mm256_i32gather_ps((const float *)jbase, _mm256_add_epi32(gidx10, _mm256_set1_epi32(c + 10)), 4);
    }
}
/* codec-1 vgather: samples are q4+t3 (7 f32, 28 B/sample), so the float-unit index stride is 7 and
 * only comps 0..6 are gathered - dropping 6 of the 20 vgathers. Scale lanes 7..9 are set to 1.0
 * (the codec-1 affine is scale-free and never reads them, but they stay well-defined). */
static inline void mrw_gather_vg_c1(const mrw_blob_f32 *jbase, __m256i gidx7, mrw_w A_[10], mrw_w B_[10]) {
    for (int c = 0; c < 7; ++c) {
        A_[c] = _mm256_i32gather_ps((const float *)jbase, _mm256_add_epi32(gidx7, _mm256_set1_epi32(c)),     4);
        B_[c] = _mm256_i32gather_ps((const float *)jbase, _mm256_add_epi32(gidx7, _mm256_set1_epi32(c + 7)), 4);
    }
    mrw_w one = mrw_w_set1(1.0f);
    A_[7]=A_[8]=A_[9]=one; B_[7]=B_[8]=B_[9]=one;
}
#endif

/* scatter the palette (M = 12 components × 8 lanes, SoA in registers) to the AoS output via an
 * in-register 8×8 (+ 8×4) transpose - no SoA staging buffer and no per-float scalar store, so the
 * SoA→AoS step is 8-wide data movement instead of the scalar deinterleave that throttled AVX2. Each
 * lane's 12-float entry lands at a stride of jc*12 floats; LIVE lanes only (padded lanes never reach
 * caller output). Full tiles (live==8) store straight from registers (the hot path); the rare
 * remainder stages 8 rows and copies the live ones. */
static inline void mrw_scatter_palette(const mrw_w M[12], void *out_palettes,
                                       mrw_palette_format out_format, int use_f16c,
                                       uint32_t row_base, uint32_t jc, uint32_t j, uint32_t live) {
    /* 8×8 transpose of comps 0..7 → r0..r7, where r_l = lane l's components 0..7. */
    __m256 a0=_mm256_unpacklo_ps(M[0],M[1]), a1=_mm256_unpackhi_ps(M[0],M[1]);
    __m256 a2=_mm256_unpacklo_ps(M[2],M[3]), a3=_mm256_unpackhi_ps(M[2],M[3]);
    __m256 a4=_mm256_unpacklo_ps(M[4],M[5]), a5=_mm256_unpackhi_ps(M[4],M[5]);
    __m256 a6=_mm256_unpacklo_ps(M[6],M[7]), a7=_mm256_unpackhi_ps(M[6],M[7]);
    __m256 b0=_mm256_shuffle_ps(a0,a2,0x44), b1=_mm256_shuffle_ps(a0,a2,0xEE);
    __m256 b2=_mm256_shuffle_ps(a1,a3,0x44), b3=_mm256_shuffle_ps(a1,a3,0xEE);
    __m256 b4=_mm256_shuffle_ps(a4,a6,0x44), b5=_mm256_shuffle_ps(a4,a6,0xEE);
    __m256 b6=_mm256_shuffle_ps(a5,a7,0x44), b7=_mm256_shuffle_ps(a5,a7,0xEE);
    __m256 r0=_mm256_permute2f128_ps(b0,b4,0x20), r4=_mm256_permute2f128_ps(b0,b4,0x31);
    __m256 r1=_mm256_permute2f128_ps(b1,b5,0x20), r5=_mm256_permute2f128_ps(b1,b5,0x31);
    __m256 r2=_mm256_permute2f128_ps(b2,b6,0x20), r6=_mm256_permute2f128_ps(b2,b6,0x31);
    __m256 r3=_mm256_permute2f128_ps(b3,b7,0x20), r7=_mm256_permute2f128_ps(b3,b7,0x31);
    /* 8×4 transpose of comps 8..11 → e0..e7, where e_l = lane l's components 8..11 (one xmm). */
    __m256 c0=_mm256_unpacklo_ps(M[8],M[9]),   c1=_mm256_unpackhi_ps(M[8],M[9]);
    __m256 c2=_mm256_unpacklo_ps(M[10],M[11]), c3=_mm256_unpackhi_ps(M[10],M[11]);
    __m256 d0=_mm256_shuffle_ps(c0,c2,0x44), d1=_mm256_shuffle_ps(c0,c2,0xEE);
    __m256 d2=_mm256_shuffle_ps(c1,c3,0x44), d3=_mm256_shuffle_ps(c1,c3,0xEE);
    __m128 e0=_mm256_castps256_ps128(d0), e4=_mm256_extractf128_ps(d0,1);
    __m128 e1=_mm256_castps256_ps128(d1), e5=_mm256_extractf128_ps(d1,1);
    __m128 e2=_mm256_castps256_ps128(d2), e6=_mm256_extractf128_ps(d2,1);
    __m128 e3=_mm256_castps256_ps128(d3), e7=_mm256_extractf128_ps(d3,1);

    if (out_format == MRW_PALETTE_F16) {
        /* Narrow the final AoS to binary16. Stage the 8 transposed rows to a
         * cache-hot AoS buffer (same as the f32 remainder), then per LIVE lane convert 12 f32 → 12 f16
         * and store 16 B (comps 0..7) + 8 B (comps 8..11). 24-B entries leave per-joint starts only
         * 8-byte aligned on odd jc, so the stores are UNALIGNED (storeu / storel) - never an aligned
         * cast. With F16C (use_f16c) the convert is vcvtps2ph (RNE); else the scalar mrw_f32_to_f16
         * fallback, which is bit-identical for these finite values. */
        _Alignas(MRW_W_ALIGN) float aos[MRW_LANES][MRW_PALETTE_FLOATS];
        _mm256_storeu_ps(aos[0], r0); _mm_storeu_ps(aos[0]+8, e0);
        _mm256_storeu_ps(aos[1], r1); _mm_storeu_ps(aos[1]+8, e1);
        _mm256_storeu_ps(aos[2], r2); _mm_storeu_ps(aos[2]+8, e2);
        _mm256_storeu_ps(aos[3], r3); _mm_storeu_ps(aos[3]+8, e3);
        _mm256_storeu_ps(aos[4], r4); _mm_storeu_ps(aos[4]+8, e4);
        _mm256_storeu_ps(aos[5], r5); _mm_storeu_ps(aos[5]+8, e5);
        _mm256_storeu_ps(aos[6], r6); _mm_storeu_ps(aos[6]+8, e6);
        _mm256_storeu_ps(aos[7], r7); _mm_storeu_ps(aos[7]+8, e7);
        uint16_t *base16 = (uint16_t *)out_palettes + ((size_t)row_base * jc + j) * MRW_PALETTE_FLOATS;
        const size_t s16 = (size_t)jc * MRW_PALETTE_FLOATS; /* per-instance stride (halfs) */
        for (uint32_t l = 0; l < live; ++l) {
            uint16_t *d16 = base16 + (size_t)l * s16;
            if (use_f16c) {
                /* RNE; the rounding mode is a compile-time immediate. _MM_FROUND_TO_NEAREST_INT only
                 * (no _MM_FROUND_NO_EXC): the bits are identical either way - NO_EXC only suppresses the
                 * inexact flag the kernel's own arithmetic already raises - and it keeps the immediate in
                 * MSVC's accepted 0..7 range (avoids C4556). */
                __m128i h8 = _mm256_cvtps_ph(_mm256_loadu_ps(aos[l]),     _MM_FROUND_TO_NEAREST_INT); /* comps 0..7  */
                __m128i h4 = _mm_cvtps_ph   (_mm_loadu_ps   (aos[l] + 8), _MM_FROUND_TO_NEAREST_INT); /* comps 8..11 */
                _mm_storeu_si128((__m128i *)(void *)d16, h8);                   /* 16 B unaligned */
                _mm_storel_epi64((__m128i *)(void *)(d16 + 8), h4);            /*  8 B unaligned */
            } else {
                for (int c = 0; c < MRW_PALETTE_FLOATS; ++c) d16[c] = mrw_f32_to_f16(aos[l][c]);
            }
        }
        return;
    }

    float *d = (float *)out_palettes + ((size_t)row_base * jc + j) * MRW_PALETTE_FLOATS;
    const size_t s = (size_t)jc * MRW_PALETTE_FLOATS; /* per-instance stride (floats) */
    if (live == MRW_LANES) { /* full tile: straight-line strided stores, no staging */
        _mm256_storeu_ps(d,         r0); _mm_storeu_ps(d         + 8, e0);
        _mm256_storeu_ps(d +   s,   r1); _mm_storeu_ps(d +   s   + 8, e1);
        _mm256_storeu_ps(d + 2*s,   r2); _mm_storeu_ps(d + 2*s   + 8, e2);
        _mm256_storeu_ps(d + 3*s,   r3); _mm_storeu_ps(d + 3*s   + 8, e3);
        _mm256_storeu_ps(d + 4*s,   r4); _mm_storeu_ps(d + 4*s   + 8, e4);
        _mm256_storeu_ps(d + 5*s,   r5); _mm_storeu_ps(d + 5*s   + 8, e5);
        _mm256_storeu_ps(d + 6*s,   r6); _mm_storeu_ps(d + 6*s   + 8, e6);
        _mm256_storeu_ps(d + 7*s,   r7); _mm_storeu_ps(d + 7*s   + 8, e7);
    } else {
        _Alignas(MRW_W_ALIGN) float aos[MRW_LANES][MRW_PALETTE_FLOATS];
        _mm256_storeu_ps(aos[0], r0); _mm_storeu_ps(aos[0]+8, e0);
        _mm256_storeu_ps(aos[1], r1); _mm_storeu_ps(aos[1]+8, e1);
        _mm256_storeu_ps(aos[2], r2); _mm_storeu_ps(aos[2]+8, e2);
        _mm256_storeu_ps(aos[3], r3); _mm_storeu_ps(aos[3]+8, e3);
        _mm256_storeu_ps(aos[4], r4); _mm_storeu_ps(aos[4]+8, e4);
        _mm256_storeu_ps(aos[5], r5); _mm_storeu_ps(aos[5]+8, e5);
        _mm256_storeu_ps(aos[6], r6); _mm_storeu_ps(aos[6]+8, e6);
        _mm256_storeu_ps(aos[7], r7); _mm_storeu_ps(aos[7]+8, e7);
        for (uint32_t l = 0; l < live; ++l)
            memcpy((float *)out_palettes + ((size_t)(row_base + l) * jc + j) * MRW_PALETTE_FLOATS,
                   aos[l], MRW_PALETTE_FLOATS * sizeof(float));
    }
}

#endif /* MRW_ISA_AVX2: vgather + scatter */

/* scalar-pack gather: read each lane's two bracket samples inline (memcpy from the blob - identical
 * bytes to mrw_clip_sample but with no external call) into DISTINCT per-lane A[k]/B[k] (so MSVC /O2
 * cannot collapse them to one sample), then pack lane-addressable wide registers via .f[k]. This is
 * SSE2's deinterleave (no hardware gather); AVX2 can also select it via MRW_AVX2_GATHER_SCALAR. */
#if !defined(MRW_ISA_AVX2) || defined(MRW_AVX2_GATHER_SCALAR)
/* Pack MRW_W per-lane bracket TRS (already decoded, scale-filled) into the 10 wide A_/B_ components. */
static inline void mrw_pack_trs(const mrw_trs *A, const mrw_trs *B, mrw_w A_[10], mrw_w B_[10]) {
    mrw_wu ax, ay, az, aw, atx, aty, atz, asx, asy, asz;
    mrw_wu bx, by, bz, bw, btx, bty, btz, bsx, bsy, bsz;
    for (uint32_t k = 0; k < MRW_W; ++k) {
        ax.f[k]=A[k].rot[0]; ay.f[k]=A[k].rot[1]; az.f[k]=A[k].rot[2]; aw.f[k]=A[k].rot[3];
        atx.f[k]=A[k].trans[0]; aty.f[k]=A[k].trans[1]; atz.f[k]=A[k].trans[2];
        asx.f[k]=A[k].scale[0]; asy.f[k]=A[k].scale[1]; asz.f[k]=A[k].scale[2];
        bx.f[k]=B[k].rot[0]; by.f[k]=B[k].rot[1]; bz.f[k]=B[k].rot[2]; bw.f[k]=B[k].rot[3];
        btx.f[k]=B[k].trans[0]; bty.f[k]=B[k].trans[1]; btz.f[k]=B[k].trans[2];
        bsx.f[k]=B[k].scale[0]; bsy.f[k]=B[k].scale[1]; bsz.f[k]=B[k].scale[2];
    }
    A_[0]=ax.v; A_[1]=ay.v; A_[2]=az.v; A_[3]=aw.v;
    A_[4]=atx.v; A_[5]=aty.v; A_[6]=atz.v; A_[7]=asx.v; A_[8]=asy.v; A_[9]=asz.v;
    B_[0]=bx.v; B_[1]=by.v; B_[2]=bz.v; B_[3]=bw.v;
    B_[4]=btx.v; B_[5]=bty.v; B_[6]=btz.v; B_[7]=bsx.v; B_[8]=bsy.v; B_[9]=bsz.v;
}
static inline void mrw_gather_sc(const uint8_t *jbase, const uint32_t *i0, uint32_t L,
                                 mrw_w A_[10], mrw_w B_[10]) {
    mrw_trs A[MRW_W], B[MRW_W];
    for (uint32_t k = 0; k < MRW_W; ++k) {
        uint32_t f = i0[L + k];
        memcpy(&A[k], jbase + (size_t)f       * MRW_CLIP_SAMPLE_STRIDE, MRW_CLIP_SAMPLE_STRIDE);
        memcpy(&B[k], jbase + (size_t)(f + 1) * MRW_CLIP_SAMPLE_STRIDE, MRW_CLIP_SAMPLE_STRIDE);
    }
    mrw_pack_trs(A, B, A_, B_);
}
/* codec-1 scalar gather: samples are q4+t3 (28 B); read 28 B per bracket and set scale := (1,1,1). */
static inline void mrw_gather_sc_c1(const uint8_t *jbase, const uint32_t *i0, uint32_t L,
                                    mrw_w A_[10], mrw_w B_[10]) {
    mrw_trs A[MRW_W], B[MRW_W];
    for (uint32_t k = 0; k < MRW_W; ++k) {
        uint32_t f = i0[L + k];
        memcpy(&A[k], jbase + (size_t)f       * MRW_ROOT_SAMPLE_STRIDE, MRW_ROOT_SAMPLE_STRIDE);
        memcpy(&B[k], jbase + (size_t)(f + 1) * MRW_ROOT_SAMPLE_STRIDE, MRW_ROOT_SAMPLE_STRIDE);
        A[k].scale[0]=A[k].scale[1]=A[k].scale[2]=1.0f;
        B[k].scale[0]=B[k].scale[1]=B[k].scale[2]=1.0f;
    }
    mrw_pack_trs(A, B, A_, B_);
}
#endif

#if defined(MRW_BENCH_NO_SCATTER)
/* Compute-only benchmark floor: the real kernel MINUS the palette write. Instead of scattering M_j
 * to the AoS output, accumulate all 12 components into a thread-local 8-float park so the inverse-
 * bind fold (and the gather/nlerp/compose feeding it) stay live, with no DRAM store. The RAW chain
 * on the park defeats dead-store elimination; one L1-resident line per thread ⇒ no false sharing.
 * Bench-only: NEVER defined in the runtime build. */
#if defined(_MSC_VER)
#  define MRW_BENCH_TLS __declspec(thread)
#else
#  define MRW_BENCH_TLS _Thread_local
#endif
static MRW_BENCH_TLS _Alignas(32) float mrw_bench_park[8];
static inline void mrw_bench_consume(const mrw_w Mj[12]) {
    mrw_w acc = mrw_w_load(mrw_bench_park);
    for (int c = 0; c < 12; ++c) acc = mrw_w_add(acc, Mj[c]);
    mrw_w_store(mrw_bench_park, acc);
}
#endif

/* ---- additional wide helpers for the pose-combine kernels (blend / accumulate) ---- */

/* Clamp each lane to [0,1] via select (no min/max intrinsic needed). Kernel inputs are finite-
 * validated upstream, so no NaN reaches here (the scalar clampf has the same no-NaN precondition). */
static inline mrw_w mrw_w_clamp01(mrw_w x) {
    mrw_w one = mrw_w_set1(1.0f), zero = mrw_w_zero();
    x = mrw_w_select(mrw_w_cmpgt(x, one), one, x);    /* x > 1 → 1 */
    x = mrw_w_select(mrw_w_cmpgt(zero, x), zero, x);  /* x < 0 → 0 */
    return x;
}

/* Hamilton product o = a ⊗ b per `wide` lane (mirrors scalar mrw_quat_mul ordering).
 * Outputs distinct from inputs. */
static inline void mrw_quat_mul_w(mrw_w ax, mrw_w ay, mrw_w az, mrw_w aw,
                                  mrw_w bx, mrw_w by, mrw_w bz, mrw_w bw,
                                  mrw_w *ox, mrw_w *oy, mrw_w *oz, mrw_w *ow) {
    *ox = mrw_w_sub(mrw_w_add(mrw_w_add(mrw_w_mul(aw,bx), mrw_w_mul(ax,bw)), mrw_w_mul(ay,bz)), mrw_w_mul(az,by));
    *oy = mrw_w_add(mrw_w_add(mrw_w_sub(mrw_w_mul(aw,by), mrw_w_mul(ax,bz)), mrw_w_mul(ay,bw)), mrw_w_mul(az,bx));
    *oz = mrw_w_add(mrw_w_sub(mrw_w_add(mrw_w_mul(aw,bz), mrw_w_mul(ax,by)), mrw_w_mul(ay,bx)), mrw_w_mul(az,bw));
    *ow = mrw_w_sub(mrw_w_sub(mrw_w_sub(mrw_w_mul(aw,bw), mrw_w_mul(ax,bx)), mrw_w_mul(ay,by)), mrw_w_mul(az,bz));
}

/* Normalize a `wide` quaternion in place; n2 ≤ 0 → 0 (the zero-fallback, as in mrw_quat_nlerp_w). */
static inline void mrw_quat_normalize_w(mrw_w *x, mrw_w *y, mrw_w *z, mrw_w *w) {
    mrw_w n2  = mrw_w_fma(*x,*x, mrw_w_fma(*y,*y, mrw_w_fma(*z,*z, mrw_w_mul(*w,*w))));
    mrw_w inv = mrw_w_div(mrw_w_set1(1.0f), mrw_w_sqrt(n2));
    inv = mrw_w_select(mrw_w_cmpgt(n2, mrw_w_zero()), inv, mrw_w_zero());
    *x = mrw_w_mul(*x, inv); *y = mrw_w_mul(*y, inv); *z = mrw_w_mul(*z, inv); *w = mrw_w_mul(*w, inv);
}
#endif /* MRW_BATCH_KERNEL_H */
