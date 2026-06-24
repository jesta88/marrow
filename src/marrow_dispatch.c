/* marrow dispatch - caller-owned, immutable, value-typed backend selection.
 * No lazy globals. _detect picks the fastest implemented backend the host supports
 * (AVX2(+FMA) ▸ SSE2 ▸ scalar) and reports that backend's canonical feature bits. */
#include "marrow.h"
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define MRW_X86 1
#  if defined(_MSC_VER)
#    include <intrin.h>
static void mrw_cpuidex(int regs[4], int leaf, int subleaf) { __cpuidex(regs, leaf, subleaf); }
static uint64_t mrw_xgetbv0(void) { return _xgetbv(0); }
#  else
#    include <cpuid.h>
static void mrw_cpuidex(int regs[4], int leaf, int subleaf) {
    unsigned a, b, c, d;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    regs[0] = (int)a; regs[1] = (int)b; regs[2] = (int)c; regs[3] = (int)d;
}
static uint64_t mrw_xgetbv0(void) {
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}
#  endif
#else
#  define MRW_X86 0
#endif

/* Detect the host's usable feature set (CPUID AVX/AVX2, OSXSAVE+YMM, FMA
 * independently). Returns a mask of MRW_FEAT_* bits. */
static uint32_t detect_features(void) {
#if MRW_X86
    uint32_t feats = 0;
    int r[4];
    mrw_cpuidex(r, 0, 0);
    int max_leaf = r[0];
    if (max_leaf < 1) return 0;

    mrw_cpuidex(r, 1, 0);
    uint32_t ecx1 = (uint32_t)r[2], edx1 = (uint32_t)r[3];
    int has_sse2    = (edx1 >> 26) & 1;
    int has_avx_cpu = (ecx1 >> 28) & 1;
    int has_osxsave = (ecx1 >> 27) & 1;
    int has_fma_cpu = (ecx1 >> 12) & 1;
    int has_f16c_cpu = (ecx1 >> 29) & 1;

    int ymm_enabled = 0;
    if (has_osxsave) {
        uint64_t xcr0 = mrw_xgetbv0();
        ymm_enabled = ((xcr0 & 0x6u) == 0x6u); /* XMM(bit1) + YMM(bit2) state saved */
    }
    if (has_sse2) feats |= MRW_FEAT_SSE2;
    if (has_osxsave && ymm_enabled) feats |= MRW_FEAT_OSXSAVE_YMM;

    int avx_usable = has_avx_cpu && ymm_enabled;
    if (avx_usable) feats |= MRW_FEAT_AVX;
    if (has_fma_cpu && avx_usable) feats |= MRW_FEAT_FMA;   /* FMA needs YMM state */
    if (has_f16c_cpu && avx_usable) feats |= MRW_FEAT_F16C; /* vcvtps2ph is VEX-encoded → needs YMM state */

    if (max_leaf >= 7) {
        mrw_cpuidex(r, 7, 0);
        int has_avx2_cpu = ((uint32_t)r[1] >> 5) & 1;
        if (has_avx2_cpu && avx_usable) feats |= MRW_FEAT_AVX2;
    }
    return feats;
#else
    return 0;
#endif
}

mrw_result mrw_dispatch_scalar(mrw_dispatch *out) {
    if (!out) return MRW_E_RANGE;
    out->backend = MRW_BACKEND_SCALAR;
    out->features = 0;
    return MRW_OK;
}

mrw_result mrw_dispatch_detect(mrw_dispatch *out) {
    if (!out) return MRW_E_RANGE;
    /* Pick the best implemented backend the host supports: AVX2 (+FMA if present) > SSE2 > scalar.
     * The carried feature bits mirror the forced constructors so the batch dispatcher can route
     * AVX2-with-FMA vs AVX2-without-FMA. */
    uint32_t feats = detect_features();
    uint32_t avx2_need = MRW_FEAT_AVX2 | MRW_FEAT_OSXSAVE_YMM;
    if ((feats & avx2_need) == avx2_need) {
        out->backend = MRW_BACKEND_AVX2;
        out->features = MRW_FEAT_SSE2 | MRW_FEAT_AVX | MRW_FEAT_AVX2 | MRW_FEAT_OSXSAVE_YMM
                      | (feats & MRW_FEAT_FMA) | (feats & MRW_FEAT_F16C);
    } else if (feats & MRW_FEAT_SSE2) {
        out->backend = MRW_BACKEND_SSE2;
        out->features = MRW_FEAT_SSE2;
    } else {
        out->backend = MRW_BACKEND_SCALAR;
        out->features = 0; /* canonical scalar dispatch (no SIMD backend usable on this host) */
    }
    return MRW_OK;
}

mrw_result mrw_dispatch_sse2(mrw_dispatch *out) {
    if (!out) return MRW_E_RANGE;
    uint32_t feats = detect_features();
    if (!(feats & MRW_FEAT_SSE2)) { out->backend = 0; out->features = 0; return MRW_E_UNSUPPORTED; }
    out->backend = MRW_BACKEND_SSE2;
    out->features = MRW_FEAT_SSE2;
    return MRW_OK;
}

mrw_result mrw_dispatch_avx2(mrw_dispatch *out) {
    if (!out) return MRW_E_RANGE;
    uint32_t feats = detect_features();
    uint32_t need = MRW_FEAT_AVX2 | MRW_FEAT_OSXSAVE_YMM;
    if ((feats & need) != need) { out->backend = 0; out->features = 0; return MRW_E_UNSUPPORTED; }
    out->backend = MRW_BACKEND_AVX2;
    /* AVX2 implies AVX/SSE2/YMM; carry FMA and F16C through only if the host actually has them. */
    out->features = MRW_FEAT_SSE2 | MRW_FEAT_AVX | MRW_FEAT_AVX2 | MRW_FEAT_OSXSAVE_YMM
                  | (feats & MRW_FEAT_FMA) | (feats & MRW_FEAT_F16C);
    return MRW_OK;
}
