/* marrow wide - compile-time-monomorphized vertical SIMD for the across-instance SoA batch
 * scratch. Each ISA translation unit defines exactly one of MRW_ISA_SSE2 / MRW_ISA_AVX2 (and,
 * for AVX2, optionally MRW_ISA_FMA) before including this header; every wrapper is `static
 * inline` and lowers to that ISA's exact instruction, so there is no runtime polymorphism and
 * no lowest-common-denominator penalty.
 *
 * VERTICAL OPS ONLY. The lane axis is across-instance (lane = instance), so the per-joint pose
 * math is component-wise with no cross-128-lane movement - which is why
 * one source serves SSE2 and AVX2 with no AVX2 compromise. The lane-crossing data movement
 * (per-lane keyframe gather, SoA→AoS scatter) is deliberately NOT here; it stays per-ISA in
 * the kernel. Keep it that way: adding a shuffle/permute/gather wrapper here is the smell that
 * the seam moved into the shared path. */
#ifndef MRW_WIDE_H
#define MRW_WIDE_H

#if defined(MRW_ISA_AVX2)
  #include <immintrin.h>
  typedef __m256 mrw_w;
  #define MRW_W 8u
  #define MRW_W_ALIGN 32   /* alignment for an MRW_W-float array feeding mrw_w_load */
  static inline mrw_w mrw_w_set1 (float x)            { return _mm256_set1_ps(x); }
  static inline mrw_w mrw_w_zero (void)               { return _mm256_setzero_ps(); }
  static inline mrw_w mrw_w_load (const float *p)     { return _mm256_load_ps(p); }   /* 32-byte aligned */
  static inline mrw_w mrw_w_loadu(const float *p)     { return _mm256_loadu_ps(p); }
  static inline void  mrw_w_store (float *p, mrw_w v) { _mm256_store_ps(p, v); }
  static inline mrw_w mrw_w_add(mrw_w a, mrw_w b)     { return _mm256_add_ps(a, b); }
  static inline mrw_w mrw_w_sub(mrw_w a, mrw_w b)     { return _mm256_sub_ps(a, b); }
  static inline mrw_w mrw_w_mul(mrw_w a, mrw_w b)     { return _mm256_mul_ps(a, b); }
  #if defined(MRW_ISA_FMA)
    static inline mrw_w mrw_w_fma(mrw_w a, mrw_w b, mrw_w c) { return _mm256_fmadd_ps(a, b, c); } /* a*b + c, fused */
  #else
    static inline mrw_w mrw_w_fma(mrw_w a, mrw_w b, mrw_w c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
  #endif
  static inline mrw_w mrw_w_div (mrw_w a, mrw_w b)    { return _mm256_div_ps(a, b); }
  static inline mrw_w mrw_w_sqrt(mrw_w a)            { return _mm256_sqrt_ps(a); }
  static inline mrw_w mrw_w_and   (mrw_w a, mrw_w b)  { return _mm256_and_ps(a, b); }
  static inline mrw_w mrw_w_xor   (mrw_w a, mrw_w b)  { return _mm256_xor_ps(a, b); }
  static inline mrw_w mrw_w_andnot(mrw_w a, mrw_w b)  { return _mm256_andnot_ps(a, b); } /* (~a) & b */
  static inline mrw_w mrw_w_cmpgt (mrw_w a, mrw_w b)  { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
  /* select per lane: mask lane all-ones → t, all-zero → f. */
  static inline mrw_w mrw_w_select(mrw_w mask, mrw_w t, mrw_w f) { return _mm256_blendv_ps(f, t, mask); }

#elif defined(MRW_ISA_SSE2)
  #include <emmintrin.h>
  typedef __m128 mrw_w;
  #define MRW_W 4u
  #define MRW_W_ALIGN 16   /* alignment for an MRW_W-float array feeding mrw_w_load */
  static inline mrw_w mrw_w_set1 (float x)            { return _mm_set1_ps(x); }
  static inline mrw_w mrw_w_zero (void)               { return _mm_setzero_ps(); }
  static inline mrw_w mrw_w_load (const float *p)     { return _mm_load_ps(p); }       /* 16-byte aligned */
  static inline mrw_w mrw_w_loadu(const float *p)     { return _mm_loadu_ps(p); }
  static inline void  mrw_w_store (float *p, mrw_w v) { _mm_store_ps(p, v); }
  static inline mrw_w mrw_w_add(mrw_w a, mrw_w b)     { return _mm_add_ps(a, b); }
  static inline mrw_w mrw_w_sub(mrw_w a, mrw_w b)     { return _mm_sub_ps(a, b); }
  static inline mrw_w mrw_w_mul(mrw_w a, mrw_w b)     { return _mm_mul_ps(a, b); }
  static inline mrw_w mrw_w_fma(mrw_w a, mrw_w b, mrw_w c) { return _mm_add_ps(_mm_mul_ps(a, b), c); } /* no FMA on SSE2 */
  static inline mrw_w mrw_w_div (mrw_w a, mrw_w b)    { return _mm_div_ps(a, b); }
  static inline mrw_w mrw_w_sqrt(mrw_w a)            { return _mm_sqrt_ps(a); }
  static inline mrw_w mrw_w_and   (mrw_w a, mrw_w b)  { return _mm_and_ps(a, b); }
  static inline mrw_w mrw_w_xor   (mrw_w a, mrw_w b)  { return _mm_xor_ps(a, b); }
  static inline mrw_w mrw_w_andnot(mrw_w a, mrw_w b)  { return _mm_andnot_ps(a, b); }
  static inline mrw_w mrw_w_cmpgt (mrw_w a, mrw_w b)  { return _mm_cmpgt_ps(a, b); }
  /* SSE2 has no blendv (SSE4.1); select via mask bitops: (mask & t) | (~mask & f). */
  static inline mrw_w mrw_w_select(mrw_w mask, mrw_w t, mrw_w f) {
      return _mm_or_ps(_mm_and_ps(mask, t), _mm_andnot_ps(mask, f));
  }

#else
  #error "marrow_wide.h: define MRW_ISA_SSE2 or MRW_ISA_AVX2 before including"
#endif

/* Lane-addressable wide register: write the MRW_W scalars via .f[k], then read .v as one aligned
 * vector. This is the robust way to build a `wide` from gathered scalars - the typed .v read lowers
 * to a clean (v)movaps. (MSVC /O2 otherwise reassembles a `loadu` of a separate float[] from indexed
 * scalars and mis-orders the lanes.) Used only at the gather seam, not in the vertical hot path. */
typedef union { mrw_w v; float f[MRW_W]; } mrw_wu;

#endif /* MRW_WIDE_H */
