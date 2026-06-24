/* Homogeneous batch: randomized parity of mrw_batch_clip_to_palette against
 * the single-instance mrw_clip_to_palette reference across bone counts, tail counts, and clip
 * kinds; plus the memory/dispatch/error contract and output-canary (no padded-lane overwrite). */
#include "bench_rig.h"
#include <stdlib.h>
#include <math.h>

/* AddressSanitizer detection (nested form so __has_feature is never evaluated where it's undefined,
 * e.g. GCC). __SANITIZE_ADDRESS__ covers MSVC + GCC; __has_feature covers Clang. */
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define MRW_HAVE_ASAN 1
#  endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#  define MRW_HAVE_ASAN 1
#endif

/* Tail counts straddling every MRW_LANES (8) boundary, plus a few larger. */
static const uint32_t TAILS[] = { 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64,100 };

/* ½ ULP of binary16 at magnitude |v| - the max rounding error of an f32→f16 narrowing (the f16
 * palette parity slack added to the f32 SIMD tolerance).
 * Subnormal halves (|v| < 2^-14) have a fixed ULP of 2^-24, so the binade is clamped at -14. */
static double f16_half_ulp(double v) {
    int e; (void)frexp(fabs(v), &e);   /* |v| = m·2^e, m∈[0.5,1) ⇒ binade exponent = e-1 */
    int be = e - 1; if (be < -14) be = -14;
    return 0.5 * ldexp(1.0, be - 10);
}

/* Parity bound for the SIMD backends vs the scalar reference. The SIMD kernels use FMA and a
 * different op order, so they are NOT bit-exact (determinism is visual-only). atol+rtol is
 * ~1000× looser than FMA/reassociation noise on a 20-bone chain yet far tighter than any logic
 * bug (wrong lane/index/sign yields O(1) error). The scalar backend reuses the same ops/order
 * as the reference, so it stays bit-exact. */
#define SIMD_ATOL 1e-4
#define SIMD_RTOL 1e-3

typedef struct { mrw_dispatch d; const char *name; int simd; } backend_t;

/* Every implemented backend the host supports (SIMD ones skipped where the host lacks the ISA). */
static int host_backends(backend_t *bks) {
    int n = 0; mrw_dispatch t;
    mrw_dispatch_scalar(&t);              bks[n++] = (backend_t){ t, "scalar", 0 };
    if (mrw_dispatch_sse2(&t) == MRW_OK)  bks[n++] = (backend_t){ t, "sse2",   1 };
    if (mrw_dispatch_avx2(&t) == MRW_OK) {
        bks[n++] = (backend_t){ t, "avx2", 1 };
        /* Every real AVX2 CPU also has FMA, so the "avx2" entry above routes to the AVX2+FMA TU.
         * Force a no-FMA AVX2 dispatch too (clear the FMA bit - still a self-consistent, accepted
         * dispatch) so the base mrw_batch_kernel_avx2 is parity-tested, not shipped untested. */
        if (t.features & MRW_FEAT_FMA) {
            mrw_dispatch nf = t; nf.features &= ~(uint32_t)MRW_FEAT_FMA;
            bks[n++] = (backend_t){ nf, "avx2-nofma", 1 };
        }
        /* Force an AVX2 dispatch with F16C cleared (still self-consistent/accepted) so the f16 store's
         * scalar mrw_f32_to_f16 FALLBACK in the AVX2 TU is parity-tested, not just the vcvtps2ph path. */
        if (t.features & MRW_FEAT_F16C) {
            mrw_dispatch n16 = t; n16.features &= ~(uint32_t)MRW_FEAT_F16C;
            bks[n++] = (backend_t){ n16, "avx2-nof16c", 1 };
        }
    }
    return n;
}

/* Build a rig, then for every tail count check each backend's batch == loop(reference) (scalar
 * bit-exact, SIMD within tolerance) and that the scatter never writes past the required output. */
static void run_parity(uint32_t jc, uint32_t sc, uint32_t flags, uint32_t seed, uint32_t codec) {
    uint8_t *buf = NULL;
    size_t sz = build_rig_blob_codec(jc, sc, flags, seed, codec, &buf);
    mrw_blob blob; CHECK_EQ(mrw_blob_open(buf, sz, &blob), MRW_OK);
    mrw_skeleton_view sv; CHECK_EQ(mrw_blob_skeleton(&blob, &sv), MRW_OK);
    mrw_clip_view cv;     CHECK_EQ(mrw_clip_view_at(&blob, 1, &cv), MRW_OK);

    backend_t bks[6]; int nbk = host_backends(bks);
    float dur = (sc > 1) ? (float)(sc - 1) / RIG_FPS : 0.0f;
    rig_rng r = { seed * 2654435761u + 7u };

    for (size_t ti = 0; ti < sizeof TAILS / sizeof TAILS[0]; ++ti) {
        uint32_t N = TAILS[ti];

        mrw_mem_req sreq, oreq;
        CHECK_EQ(mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F32, &sreq, &oreq), MRW_OK);
        CHECK_EQ(sreq.align, 64);
        CHECK_EQ(oreq.align, 16);
        CHECK_EQ(oreq.size, (size_t)N * jc * 12u * sizeof(float));

        float *times = (float *)malloc((size_t)N * sizeof(float));
        for (uint32_t i = 0; i < N; ++i)
            times[i] = (sc > 1) ? rig_f(&r, -1.5f * dur, 1.5f * dur) : rig_f(&r, -5.0f, 5.0f);

        size_t obytes = oreq.size, cap = obytes + 64; /* 64 sentinel bytes past the end */
        uint8_t *scr = mrw_alloc64(sreq.size ? sreq.size : 64);

        for (int b = 0; b < nbk; ++b) {
            uint8_t *outb = mrw_alloc64(cap);
            memset(outb, 0xAB, cap);
            float *out = (float *)outb;

            CHECK_EQ(mrw_batch_clip_to_palette(&bks[b].d, &sv, &cv, times, N, out, obytes, scr, sreq.size), MRW_OK);

            float refscr[RIG_MAX_JOINTS * 12], refpal[RIG_MAX_JOINTS * 12];
            for (uint32_t i = 0; i < N; ++i) {
                CHECK_EQ(mrw_clip_to_palette(&sv, &cv, times[i], refscr, refpal, jc), MRW_OK);
                for (uint32_t k = 0; k < jc * 12u; ++k) {
                    double a = (double)out[(size_t)i * jc * 12u + k], e = (double)refpal[k];
                    double diff = a - e; if (diff < 0) diff = -diff;
                    double tol = bks[b].simd ? (SIMD_ATOL + SIMD_RTOL * (e < 0 ? -e : e)) : 1e-6;
                    if (diff > tol) {
                        printf("FAIL parity[%s] codec=%u sc=%u fl=%u jc=%u N=%u i=%u j=%u c=%u: |%g - %g| = %g > %g\n",
                               bks[b].name, codec, sc, flags, jc, N, i, k/12u, k%12u, a, e, diff, tol);
                        ++g_fail; break;
                    }
                }
            }
            for (size_t k = obytes; k < cap; ++k)
                if (outb[k] != 0xAB) { printf("FAIL canary[%s] jc=%u N=%u byte %zu\n", bks[b].name, jc, N, k); ++g_fail; break; }

            /* f16 output. Two gates:
             *  (1) bit-exact internal consistency - the f16 store IS the f32 store narrowed, so for
             *      EVERY backend out16[k] == mrw_f32_to_f16(out[k]) (catches an f16-specific
             *      transpose/lane bug, and verifies the AVX2 vcvtps2ph path == the scalar encoder);
             *  (2) end-to-end precision - decode(out16[k]) within (f32 SIMD tol + ½ f16 ULP) of the
             *      scalar f32 reference. */
            mrw_mem_req oreq16;
            CHECK_EQ(mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F16, NULL, &oreq16), MRW_OK);
            CHECK_EQ(oreq16.align, 16);
            CHECK_EQ(oreq16.size, (size_t)N * jc * 12u * sizeof(uint16_t));
            size_t cap16 = oreq16.size + 64;
            uint8_t *out16b = mrw_alloc64(cap16);
            memset(out16b, 0xAB, cap16);
            uint16_t *out16 = (uint16_t *)out16b;
            CHECK_EQ(mrw_batch_clip_to_palette_f16(&bks[b].d, &sv, &cv, times, N, out16, oreq16.size, scr, sreq.size), MRW_OK);
            for (uint32_t i = 0; i < N && !g_fail; ++i) {
                CHECK_EQ(mrw_clip_to_palette(&sv, &cv, times[i], refscr, refpal, jc), MRW_OK);
                for (uint32_t k = 0; k < jc * 12u; ++k) {
                    size_t idx = (size_t)i * jc * 12u + k;
                    uint16_t h = out16[idx], hx = mrw_f32_to_f16(out[idx]);
                    if (h != hx) {
                        printf("FAIL f16-consistency[%s] jc=%u N=%u i=%u j=%u c=%u: 0x%04X != encode(%g)=0x%04X\n",
                               bks[b].name, jc, N, i, k/12u, k%12u, h, out[idx], hx); ++g_fail; break;
                    }
                    double a = (double)mrw_half_to_float(h), e = (double)refpal[k];
                    double diff = a - e; if (diff < 0) diff = -diff;
                    double tol = (bks[b].simd ? (SIMD_ATOL + SIMD_RTOL * (e < 0 ? -e : e)) : 1e-6) + f16_half_ulp(e);
                    if (diff > tol) {
                        printf("FAIL f16-parity[%s] sc=%u jc=%u N=%u i=%u j=%u c=%u: |%g - %g| = %g > %g\n",
                               bks[b].name, sc, jc, N, i, k/12u, k%12u, a, e, diff, tol); ++g_fail; break;
                    }
                }
            }
            for (size_t k = oreq16.size; k < cap16; ++k)
                if (out16b[k] != 0xAB) { printf("FAIL f16-canary[%s] jc=%u N=%u byte %zu\n", bks[b].name, jc, N, k); ++g_fail; break; }
            mrw_free(out16b);
            mrw_free(outb);
        }
        free(times); mrw_free(scr);
    }
    mrw_free(buf);
}

/* Non-8-aligned-chunk parity - the per-call correctness a multi-worker partition relies on.
 * A job pool splits [0,N) at arbitrary boundaries that need NOT round to MRW_LANES, so a worker
 * writes into a non-MRW_LANES-aligned output offset and its first/last tiles land on cache lines
 * shared with the neighbouring range. This is the SEQUENTIAL precondition for that to be safe:
 * write [0,s) then [s,N) as TWO batch calls into ONE buffer at a non-8-aligned split s and a
 * 16-byte-but-not-64-byte-aligned base (so the non-temporal store path's 16-vs-32 phase lead-in
 * fires for even jc too), and prove (a) each call writes ONLY its own [begin,end) instances -
 * checked by a fill-canary sweep of the untouched tail right after the first call, so a stray
 * lead-in/tail overrun into the neighbour can't be masked when the second call overwrites it -
 * and (b) the whole result stays reference-exact. The CONCURRENT cross-core case (two workers
 * streaming disjoint instance ranges that share a boundary line) rests on the all-non-temporal,
 * disjoint-byte property - every output byte is written by exactly one call, and non-temporal
 * partial-line writes carry byte-enables - and is exercised live by the pooled bench run, not by
 * this deterministic unit. */
static void run_split_parity(uint32_t jc, uint32_t sc, uint32_t flags, uint32_t seed) {
    uint8_t *buf = NULL;
    size_t sz = build_rig_blob(jc, sc, flags, seed, &buf);
    mrw_blob blob; CHECK_EQ(mrw_blob_open(buf, sz, &blob), MRW_OK);
    mrw_skeleton_view sv; CHECK_EQ(mrw_blob_skeleton(&blob, &sv), MRW_OK);
    mrw_clip_view cv;     CHECK_EQ(mrw_clip_view_at(&blob, 1, &cv), MRW_OK);

    backend_t bks[6]; int nbk = host_backends(bks);
    float dur = (sc > 1) ? (float)(sc - 1) / RIG_FPS : 0.0f;
    rig_rng r = { seed * 2654435761u + 11u };

    const uint32_t N = 100;
    const uint32_t splits[] = { 1, 3, 7, 8, 9, 11, 13, 50 }; /* mostly non-8-aligned; 8 = aligned control */

    float *times = (float *)malloc((size_t)N * sizeof(float));
    for (uint32_t i = 0; i < N; ++i)
        times[i] = (sc > 1) ? rig_f(&r, -1.5f * dur, 1.5f * dur) : rig_f(&r, -5.0f, 5.0f);

    mrw_mem_req sreq, oreq;
    CHECK_EQ(mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F32, &sreq, &oreq), MRW_OK);
    uint8_t *scr = mrw_alloc64(sreq.size ? sreq.size : 64);
    const size_t row = (size_t)jc * 12u;                 /* floats per instance                       */

    for (size_t si = 0; si < sizeof splits / sizeof splits[0]; ++si) {
        uint32_t s = splits[si];
        for (int b = 0; b < nbk; ++b) {
            /* 16-but-not-64-aligned live base: shift 16 bytes (4 floats) off the 64-aligned block. */
            uint8_t *outb = mrw_alloc64(oreq.size + 16 + 64);
            memset(outb, 0xAB, oreq.size + 16 + 64);
            float *out = (float *)(outb + 16);
            int bad = 0;

            /* call A writes ONLY instances [0,s); the second call (and the eventual parity check) would
             * otherwise overwrite the evidence of an A overrun, so verify the whole tail past A's logical
             * end - instance s onward, incl. the 64-byte trailing canary - is still the fill BEFORE call B. */
            CHECK_EQ(mrw_batch_clip_to_palette(&bks[b].d, &sv, &cv, times, s,
                       out, oreq.size, scr, sreq.size), MRW_OK);
            for (size_t k = 16 + (size_t)s * row * sizeof(float); k < 16 + oreq.size + 64; ++k)
                if (outb[k] != 0xAB) { printf("FAIL split-Aoverrun[%s] jc=%u N=%u s=%u byte %zu\n", bks[b].name, jc, N, s, k); ++g_fail; bad = 1; break; }

            /* call B writes [s,N) at the non-8-aligned offset s·row (mimics the neighbouring worker) */
            if (!bad)
                CHECK_EQ(mrw_batch_clip_to_palette(&bks[b].d, &sv, &cv, times + s, N - s,
                           out + (size_t)s * row, oreq.size - (size_t)s * row * sizeof(float),
                           scr, sreq.size), MRW_OK);

            float refscr[RIG_MAX_JOINTS * 12], refpal[RIG_MAX_JOINTS * 12];
            for (uint32_t i = 0; i < N && !bad; ++i) {
                CHECK_EQ(mrw_clip_to_palette(&sv, &cv, times[i], refscr, refpal, jc), MRW_OK);
                for (uint32_t k = 0; k < jc * 12u; ++k) {
                    double a = (double)out[(size_t)i * row + k], e = (double)refpal[k];
                    double diff = a - e; if (diff < 0) diff = -diff;
                    double tol = bks[b].simd ? (SIMD_ATOL + SIMD_RTOL * (e < 0 ? -e : e)) : 1e-6;
                    if (diff > tol) {
                        printf("FAIL split[%s] sc=%u fl=%u jc=%u N=%u s=%u i=%u j=%u c=%u: |%g - %g| = %g > %g\n",
                               bks[b].name, sc, flags, jc, N, s, i, k/12u, k%12u, a, e, diff, tol);
                        ++g_fail; bad = 1; break;
                    }
                }
            }
            /* canaries: the 16 bytes before the live base and the 64 bytes past the logical output end */
            for (size_t k = 0; k < 16 && !bad; ++k)
                if (outb[k] != 0xAB) { printf("FAIL split-canary-lo[%s] jc=%u s=%u byte %zu\n", bks[b].name, jc, s, k); ++g_fail; break; }
            for (size_t k = 16 + oreq.size; k < 16 + oreq.size + 64 && !bad; ++k)
                if (outb[k] != 0xAB) { printf("FAIL split-canary-hi[%s] jc=%u s=%u byte %zu\n", bks[b].name, jc, s, k); ++g_fail; break; }
            mrw_free(outb);
        }
    }
    free(times); mrw_free(scr); mrw_free(buf);
}

/* f16 sub-range parity (crowd path). The f16 entry is
 * 24 B/joint, so on ODD joint counts a sub-range starting at an odd instance lands at an 8-byte- (not
 * 16-byte-) aligned output pointer - exactly what a clip-group / job-lane split produces when the
 * crowd batch writes disjoint sub-ranges of ONE tight palette in place. The scatter is fully unaligned,
 * so this MUST be accepted (not MRW_E_ALIGN) and stay reference-exact. Writes [0,s) then [s,N)
 * into one buffer at the tight 24-B stride and checks (a) both calls return MRW_OK, (b) decode parity,
 * (c) the second call's misaligned lead-in never clobbers the first range's tail or the canaries. */
static void run_split_parity_f16(uint32_t jc, uint32_t sc, uint32_t flags, uint32_t seed) {
    uint8_t *buf = NULL;
    size_t sz = build_rig_blob(jc, sc, flags, seed, &buf);
    mrw_blob blob; CHECK_EQ(mrw_blob_open(buf, sz, &blob), MRW_OK);
    mrw_skeleton_view sv; CHECK_EQ(mrw_blob_skeleton(&blob, &sv), MRW_OK);
    mrw_clip_view cv;     CHECK_EQ(mrw_clip_view_at(&blob, 1, &cv), MRW_OK);

    backend_t bks[6]; int nbk = host_backends(bks);
    float dur = (sc > 1) ? (float)(sc - 1) / RIG_FPS : 0.0f;
    rig_rng r = { seed * 2654435761u + 17u };

    const uint32_t N = 100;
    const uint32_t splits[] = { 1, 3, 7, 9, 11, 13, 50 };   /* odd s ⇒ 8-byte-aligned interior pointer on odd jc */

    float *times = (float *)malloc((size_t)N * sizeof(float));
    for (uint32_t i = 0; i < N; ++i)
        times[i] = (sc > 1) ? rig_f(&r, -1.5f * dur, 1.5f * dur) : rig_f(&r, -5.0f, 5.0f);

    mrw_mem_req sreq, oreq16;
    CHECK_EQ(mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F16, &sreq, &oreq16), MRW_OK);
    uint8_t *scr = mrw_alloc64(sreq.size ? sreq.size : 64);
    const size_t row = (size_t)jc * 12u;                    /* halfs per instance */

    for (size_t si = 0; si < sizeof splits / sizeof splits[0]; ++si) {
        uint32_t s = splits[si];
        for (int b = 0; b < nbk; ++b) {
            uint8_t *outb = mrw_alloc64(oreq16.size + 64);
            memset(outb, 0xAB, oreq16.size + 64);
            uint16_t *out = (uint16_t *)outb;
            int bad = 0;

            CHECK_EQ(mrw_batch_clip_to_palette_f16(&bks[b].d, &sv, &cv, times, s,
                       out, oreq16.size, scr, sreq.size), MRW_OK);
            /* [s,N) and the trailing canary must still be the fill before call B overwrites it */
            for (size_t k = (size_t)s * row * sizeof(uint16_t); k < oreq16.size + 64; ++k)
                if (outb[k] != 0xAB) { printf("FAIL f16-split-Aoverrun[%s] jc=%u N=%u s=%u byte %zu\n", bks[b].name, jc, N, s, k); ++g_fail; bad = 1; break; }

            /* call B writes [s,N) at the 24-B-stride interior offset (8-byte aligned on odd jc) */
            if (!bad)
                CHECK_EQ(mrw_batch_clip_to_palette_f16(&bks[b].d, &sv, &cv, times + s, N - s,
                           out + (size_t)s * row, oreq16.size - (size_t)s * row * sizeof(uint16_t),
                           scr, sreq.size), MRW_OK);

            float refscr[RIG_MAX_JOINTS * 12], refpal[RIG_MAX_JOINTS * 12];
            for (uint32_t i = 0; i < N && !bad; ++i) {
                CHECK_EQ(mrw_clip_to_palette(&sv, &cv, times[i], refscr, refpal, jc), MRW_OK);
                for (uint32_t k = 0; k < jc * 12u; ++k) {
                    double a = (double)mrw_half_to_float(out[(size_t)i * row + k]), e = (double)refpal[k];
                    double diff = a - e; if (diff < 0) diff = -diff;
                    double tol = (bks[b].simd ? (SIMD_ATOL + SIMD_RTOL * (e < 0 ? -e : e)) : 1e-6) + f16_half_ulp(e);
                    if (diff > tol) {
                        printf("FAIL f16-split[%s] sc=%u fl=%u jc=%u N=%u s=%u i=%u j=%u c=%u: |%g - %g| = %g > %g\n",
                               bks[b].name, sc, flags, jc, N, s, i, k/12u, k%12u, a, e, diff, tol);
                        ++g_fail; bad = 1; break;
                    }
                }
            }
            for (size_t k = oreq16.size; k < oreq16.size + 64 && !bad; ++k)
                if (outb[k] != 0xAB) { printf("FAIL f16-split-canary[%s] jc=%u s=%u byte %zu\n", bks[b].name, jc, s, k); ++g_fail; break; }
            mrw_free(outb);
        }
    }
    free(times); mrw_free(scr); mrw_free(buf);
}

#ifdef MRW_HAVE_ASAN
/* Exact-capacity scatter under ASan redzones. The +64-slack
 * canary tests above catch a stray write by SENTINEL scan, but that slack stays ADDRESSABLE - ASan stays
 * silent on a 1-byte over-scatter inside it. Allocating the output at EXACTLY the required size
 * (mrw_alloc64 → _aligned_malloc/posix_memalign, so the redzone abuts base+size) turns the scatter
 * boundary into a hardware-redzone check: any over-write past the palette end aborts with an ASan report.
 * Runs the f32 + f16 scatter for every tail count on each backend, at an even and an odd jc (odd ⇒ the
 * per-instance run alternates 16-/32-byte output phase). Only compiled under ASan. */
static void run_exact_capacity(void) {
    backend_t bks[6]; int nbk = host_backends(bks);
    uint32_t bones[] = { 8, 65 };
    for (size_t bi = 0; bi < sizeof bones / sizeof bones[0]; ++bi) {
        uint32_t jc = bones[bi];
        uint8_t *buf = NULL;
        size_t sz = build_rig_blob(jc, 5, 0, 900u + jc, &buf);
        mrw_blob blob; CHECK_EQ(mrw_blob_open(buf, sz, &blob), MRW_OK);
        mrw_skeleton_view sv; CHECK_EQ(mrw_blob_skeleton(&blob, &sv), MRW_OK);
        mrw_clip_view cv;     CHECK_EQ(mrw_clip_view_at(&blob, 1, &cv), MRW_OK);

        for (size_t ti = 0; ti < sizeof TAILS / sizeof TAILS[0]; ++ti) {
            uint32_t N = TAILS[ti];
            mrw_mem_req sreq, oreq, oreq16;
            CHECK_EQ(mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F32, &sreq, &oreq), MRW_OK);
            CHECK_EQ(mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F16, NULL, &oreq16), MRW_OK);
            float *times = (float *)malloc((size_t)N * sizeof(float));
            for (uint32_t i = 0; i < N; ++i) times[i] = 0.013f * (float)i;
            uint8_t *scr = mrw_alloc64(sreq.size ? sreq.size : 64);
            for (int b = 0; b < nbk; ++b) {
                uint8_t *out   = mrw_alloc64(oreq.size);    /* exact: redzone abuts the palette end */
                uint8_t *out16 = mrw_alloc64(oreq16.size);
                CHECK_EQ(mrw_batch_clip_to_palette    (&bks[b].d, &sv, &cv, times, N,
                           (float *)out, oreq.size, scr, sreq.size), MRW_OK);
                CHECK_EQ(mrw_batch_clip_to_palette_f16(&bks[b].d, &sv, &cv, times, N,
                           (uint16_t *)out16, oreq16.size, scr, sreq.size), MRW_OK);
                mrw_free(out); mrw_free(out16);
            }
            free(times); mrw_free(scr);
        }
        mrw_free(buf);
    }
    printf("test_batch: exact-capacity ASan redzone pass\n");
}
#endif /* MRW_HAVE_ASAN */

/* The memory/dispatch/error contract. */
static void run_edge(void) {
    uint8_t *buf = NULL;
    size_t sz = build_rig_blob(8, 5, 0, 123, &buf);
    mrw_blob blob; CHECK_EQ(mrw_blob_open(buf, sz, &blob), MRW_OK);
    mrw_skeleton_view sv; CHECK_EQ(mrw_blob_skeleton(&blob, &sv), MRW_OK);
    mrw_clip_view cv;     CHECK_EQ(mrw_clip_view_at(&blob, 1, &cv), MRW_OK);
    uint32_t jc = 8;

    mrw_dispatch d; mrw_dispatch_scalar(&d);
    mrw_mem_req sreq, oreq;
    CHECK_EQ(mrw_batch_clip_to_palette_requirements(jc, 4, MRW_PALETTE_F32, &sreq, &oreq), MRW_OK);
    uint8_t *scr  = mrw_alloc64(sreq.size);
    uint8_t *outb = mrw_alloc64(oreq.size + 64);
    float *out = (float *)outb;
    float times[4] = { 0.0f, 0.05f, 0.10f, 0.15f };

    /* instance_count==0: clean no-op, data pointers may be NULL */
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &sv, &cv, NULL, 0, NULL, 0, NULL, 0), MRW_OK);

    /* NULL required handles */
    CHECK_EQ(mrw_batch_clip_to_palette(NULL, &sv, &cv, times, 4, out, oreq.size, scr, sreq.size), MRW_E_RANGE);
    CHECK_EQ(mrw_batch_clip_to_palette(&d, NULL, &cv, times, 4, out, oreq.size, scr, sreq.size), MRW_E_RANGE);
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &sv, NULL, times, 4, out, oreq.size, scr, sreq.size), MRW_E_RANGE);
    /* NULL data with N>0 */
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &sv, &cv, NULL, 4, out, oreq.size, scr, sreq.size), MRW_E_RANGE);
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &sv, &cv, times, 4, NULL, oreq.size, scr, sreq.size), MRW_E_RANGE);
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &sv, &cv, times, 4, out, oreq.size, NULL, sreq.size), MRW_E_RANGE);

    /* dispatch self-consistency: a backend whose feature bits don't support it, or an
     * out-of-range backend, ⇒ MRW_E_UNSUPPORTED (no silent downgrade). Host-independent - a
     * *consistent* SIMD dispatch is exercised for real in run_parity via the forced constructors. */
    mrw_dispatch dbad_avx = { MRW_BACKEND_AVX2, 0 };   /* AVX2 backend, no feature bits */
    mrw_dispatch dbad_sse = { MRW_BACKEND_SSE2, 0 };   /* SSE2 backend, no SSE2 bit     */
    mrw_dispatch dbad_be  = { 99u, 0 };                /* backend out of range          */
    CHECK_EQ(mrw_batch_clip_to_palette(&dbad_avx, &sv, &cv, times, 4, out, oreq.size, scr, sreq.size), MRW_E_UNSUPPORTED);
    CHECK_EQ(mrw_batch_clip_to_palette(&dbad_sse, &sv, &cv, times, 4, out, oreq.size, scr, sreq.size), MRW_E_UNSUPPORTED);
    CHECK_EQ(mrw_batch_clip_to_palette(&dbad_be,  &sv, &cv, times, 4, out, oreq.size, scr, sreq.size), MRW_E_UNSUPPORTED);

    /* capacity */
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &sv, &cv, times, 4, out, oreq.size, scr, sreq.size - 1), MRW_E_CAPACITY);
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &sv, &cv, times, 4, out, oreq.size - 1, scr, sreq.size), MRW_E_CAPACITY);

    /* alignment: scr+16 is 16- but not 64-aligned ⇒ scratch ALIGN; out+4 is 4- not 16-aligned ⇒ output ALIGN */
    uint8_t *sbig = mrw_alloc64(sreq.size + 64);
    uint8_t *obig = mrw_alloc64(oreq.size + 64);
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &sv, &cv, times, 4, out, oreq.size, sbig + 16, sreq.size), MRW_E_ALIGN);
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &sv, &cv, times, 4, (float *)(obig + 4), oreq.size, scr, sreq.size), MRW_E_ALIGN);

    /* non-finite time ⇒ RANGE and NO output written (all-or-nothing) */
    memset(outb, 0xAB, oreq.size);
    float btimes[4] = { 0.0f, (float)NAN, 0.10f, 0.15f };
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &sv, &cv, btimes, 4, out, oreq.size, scr, sreq.size), MRW_E_RANGE);
    for (size_t k = 0; k < oreq.size; ++k)
        if (outb[k] != 0xAB) { printf("FAIL: output written despite non-finite time\n"); ++g_fail; break; }
    btimes[1] = (float)INFINITY;
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &sv, &cv, btimes, 4, out, oreq.size, scr, sreq.size), MRW_E_RANGE);

    mrw_free(sbig); mrw_free(obig); mrw_free(scr); mrw_free(outb); mrw_free(buf);

    /* incompatibility: mismatched joint_count, and matched joint_count but mismatched skeleton_id */
    uint8_t *bA = NULL, *bB = NULL, *bC = NULL;
    size_t szA = build_rig_blob(8, 5, 0, 1, &bA);   /* 8 joints  */
    size_t szB = build_rig_blob(12, 5, 0, 2, &bB);  /* 12 joints */
    size_t szC = build_rig_blob(8, 5, 0, 3, &bC);   /* 8 joints, different id */
    mrw_blob blA, blB, blC;
    CHECK_EQ(mrw_blob_open(bA, szA, &blA), MRW_OK);
    CHECK_EQ(mrw_blob_open(bB, szB, &blB), MRW_OK);
    CHECK_EQ(mrw_blob_open(bC, szC, &blC), MRW_OK);
    mrw_skeleton_view svA; mrw_blob_skeleton(&blA, &svA);
    mrw_clip_view cvB; mrw_clip_view_at(&blB, 1, &cvB);
    mrw_clip_view cvC; mrw_clip_view_at(&blC, 1, &cvC);

    mrw_mem_req sr2, or2;
    CHECK_EQ(mrw_batch_clip_to_palette_requirements(8, 4, MRW_PALETTE_F32, &sr2, &or2), MRW_OK);
    uint8_t *s2 = mrw_alloc64(sr2.size), *o2 = mrw_alloc64(or2.size);
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &svA, &cvB, times, 4, (float *)o2, or2.size, s2, sr2.size),
             MRW_E_INCOMPATIBLE); /* joint_count 8 vs 12 */
    CHECK_EQ(mrw_batch_clip_to_palette(&d, &svA, &cvC, times, 4, (float *)o2, or2.size, s2, sr2.size),
             MRW_E_INCOMPATIBLE); /* same count, different skeleton_id */
    mrw_free(s2); mrw_free(o2); mrw_free(bA); mrw_free(bB); mrw_free(bC);

    /* overflow in size math */
    mrw_mem_req hr;
    CHECK_EQ(mrw_batch_clip_to_palette_requirements(0xFFFFFFFFu, 0xFFFFFFFFu, MRW_PALETTE_F32, NULL, &hr), MRW_E_OVERFLOW);
    /* both-NULL out-params is a misuse */
    CHECK_EQ(mrw_batch_clip_to_palette_requirements(8, 4, MRW_PALETTE_F32, NULL, NULL), MRW_E_RANGE);
    /* out-of-range palette format ⇒ MRW_E_RANGE (both the requirements query and the batch call) */
    CHECK_EQ(mrw_batch_clip_to_palette_requirements(8, 4, 99u, &hr, NULL), MRW_E_RANGE);
}

int main(void) {
    { backend_t bk[6]; int n = host_backends(bk);
      printf("backends exercised:"); for (int i = 0; i < n; ++i) printf(" %s", bk[i].name); printf("\n"); }
    /* 8/12/20 = lean band; 65/67 = the heavy rig shapes (odd jc ⇒ per-instance output runs alternate
     * 16-/32-byte phase, exercising the non-temporal store path's lead-in/middle/tail). */
    uint32_t bones[] = { 8, 12, 20, 65, 67 };
    for (size_t b = 0; b < sizeof bones / sizeof bones[0]; ++b) {
        uint32_t jc = bones[b];
        /* codec 0 (raw TRS) AND codec 1 (scale-free q4+t3) - the rig is unit-scale, so codec 1 is an
         * exact representation and must give bit-identical poses (codec-0/1 parity). */
        for (uint32_t codec = 0; codec < 2; ++codec) {
            run_parity(jc, 1, 0,                 100u + jc, codec);  /* static (count==1)   */
            run_parity(jc, 5, 0,                 200u + jc, codec);  /* non-looping         */
            run_parity(jc, 8, MRW_CLIP_LOOPING,  300u + jc, codec);  /* looping             */
        }
    }

    /* Multi-worker partitioning: workers split the batch at arbitrary (non-MRW_LANES-aligned) instance
     * boundaries, so a worker writes into a non-8-aligned output offset and its first/last tiles share
     * cache lines with the neighbouring range. Exercise that boundary + the 16-/32-byte output phase. */
    run_split_parity(65, 5, 0,                700u);   /* heavy, non-looping, odd jc            */
    run_split_parity(67, 8, MRW_CLIP_LOOPING, 701u);   /* heavy, looping, odd jc               */
    run_split_parity(20, 5, 0,                702u);   /* lean, even jc                        */
    run_split_parity(65, 1, 0,                703u);   /* heavy, static clip                   */

    /* f16 sub-range parity: odd jc ⇒ odd-instance splits give an 8-byte-aligned interior output
     * pointer (the in-place crowd batch path); even jc is the always-16-aligned control. */
    run_split_parity_f16(65, 5, 0,                710u);   /* odd jc, non-looping  */
    run_split_parity_f16(67, 8, MRW_CLIP_LOOPING, 711u);   /* odd jc, looping      */
    run_split_parity_f16(20, 5, 0,                712u);   /* even jc control      */

    run_edge();

#ifdef MRW_HAVE_ASAN
    run_exact_capacity();   /* real redzone teeth at the scatter boundary */
#endif

    printf(g_fail ? "test_batch: %d FAILED\n" : "test_batch: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
