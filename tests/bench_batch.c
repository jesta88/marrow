/* Batch throughput benchmark at realistic scale. Compares the across-instance SoA batch path
 * against a naive loop of the single-instance AoS path, over 8/12/20 bones × realistic batch
 * sizes. Not a correctness test; reports ns/instance and the SoA-vs-loop ratio so the SIMD
 * kernels have a measured scalar baseline. `--smoke` runs a tiny config (the CTest variant). */

/* clock_gettime/CLOCK_MONOTONIC (the POSIX timer below) are hidden by glibc under strict
 * -std=c11; the feature macro must be defined before ANY system header (bench_rig.h pulls them
 * in transitively), so it sits above every #include. Ignored by MSVC. */
#ifndef _WIN32
#  define _POSIX_C_SOURCE 199309L
#endif

#include "bench_rig.h"
#include <stdlib.h>

/* ---- portable monotonic timer (Win32 / POSIX), so the target is not silently Windows-only ---- */
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
static double now_sec(void) {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#  include <time.h>
static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

static volatile float g_sink; /* defeat dead-code elimination */

static const char *backend_name(const mrw_dispatch *d) {
    switch (d->backend) {
        case MRW_BACKEND_AVX2: return (d->features & MRW_FEAT_FMA) ? "avx2+fma" : "avx2";
        case MRW_BACKEND_SSE2: return "sse2";
        default:               return "scalar";
    }
}

/* Time ONLY the batch path (no single-instance loop), returning ns/instance. Used by --isa to
 * compare backends head-to-head at one config. */
static double bench_batch_only(const mrw_dispatch *d, uint32_t jc, uint32_t N,
                               uint64_t target_instances, int best_of) {
    uint8_t *buf = NULL;
    size_t sz = build_rig_blob(jc, 8, MRW_CLIP_LOOPING, 4242u + jc, &buf);
    mrw_blob blob; mrw_blob_open(buf, sz, &blob);
    mrw_skeleton_view sv; mrw_blob_skeleton(&blob, &sv);
    mrw_clip_view cv;     mrw_clip_view_at(&blob, 1, &cv);

    mrw_mem_req sreq, oreq;
    mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F32, &sreq, &oreq);

    float *times = (float *)malloc((size_t)N * sizeof(float));
    float dur = (float)(8 - 1) / RIG_FPS;
    for (uint32_t i = 0; i < N; ++i) times[i] = dur * ((float)i / (float)N);

    uint8_t *outb = mrw_alloc64(oreq.size);
    uint8_t *scr  = mrw_alloc64(sreq.size);
    float *out = (float *)outb;

    uint32_t reps = (uint32_t)(target_instances / N); if (reps < 1) reps = 1;
    mrw_batch_clip_to_palette(d, &sv, &cv, times, N, out, oreq.size, scr, sreq.size); /* warm-up */

    double best = 1e30;
    for (int trial = 0; trial < best_of; ++trial) {
        double t0 = now_sec();
        for (uint32_t r = 0; r < reps; ++r)
            mrw_batch_clip_to_palette(d, &sv, &cv, times, N, out, oreq.size, scr, sreq.size);
        double t1 = now_sec();
        g_sink += out[0] + out[(size_t)(N - 1) * jc * 12u];
        if (t1 - t0 < best) best = t1 - t0;
    }
    double ns = best * 1e9 / ((double)reps * (double)N);
    free(times); mrw_free(outb); mrw_free(scr); mrw_free(buf);
    return ns;
}

/* Head-to-head per-ISA table: scalar / SSE2 / AVX2 / AVX2+FMA at the same configs, so the AVX2
 * scaling over SSE2 is directly visible (the demo HUD only shows the host's best backend). */
static void bench_isa(void) {
    mrw_dispatch dsc, dsse, davx, davxf;
    mrw_dispatch_scalar(&dsc);
    int has_sse = (mrw_dispatch_sse2(&dsse) == MRW_OK);
    int has_avx = (mrw_dispatch_avx2(&davx) == MRW_OK);
    /* AVX2 without FMA: same backend, canonical AVX2 bits minus FMA (the no-FMA kernel TU). davxf
     * keeps FMA; davx clears it. The fma column is only meaningful when the host actually has FMA
     * (every real AVX2 CPU does) - else davxf == davx (no FMA) and the column would be mislabeled. */
    davxf = davx;
    davx.features = MRW_FEAT_SSE2 | MRW_FEAT_AVX | MRW_FEAT_AVX2 | MRW_FEAT_OSXSAVE_YMM;
    int has_fma = has_avx && (davxf.features & MRW_FEAT_FMA) != 0;

    printf("marrow per-ISA batch comparison (ns/instance, ns/instance/bone, speedup vs SSE2)\n");
    printf("bones |        N |   scalar |     sse2 |     avx2 | avx2+fma |  ns/i/b(best) | avx2/sse2 | fma/sse2\n");
    printf("------+----------+----------+----------+----------+----------+---------------+-----------+---------\n");
    uint32_t bones[] = { 8, 20, 60 };
    uint32_t sizes[] = { 1024, 16384, 65536 };
    for (size_t b = 0; b < sizeof bones / sizeof bones[0]; ++b) {
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
            uint32_t jc = bones[b], N = sizes[s];
            double sc  = bench_batch_only(&dsc,  jc, N, 2000000u, 3);
            double sse = has_sse ? bench_batch_only(&dsse,  jc, N, 2000000u, 3) : 0;
            double avx = has_avx ? bench_batch_only(&davx,  jc, N, 2000000u, 3) : 0;
            double fma = has_fma ? bench_batch_only(&davxf, jc, N, 2000000u, 3) : 0;
            double best = fma > 0 ? fma : (avx > 0 ? avx : sse);
            printf("%5u | %8u | %8.2f | %8.2f | %8.2f | %8.2f | %13.4f | %8.2fx | %7.2fx\n",
                   jc, N, sc, sse, avx, fma, best / jc,
                   (sse > 0 && avx > 0) ? sse / avx : 0.0,
                   (sse > 0 && fma > 0) ? sse / fma : 0.0);
        }
        printf("------+----------+----------+----------+----------+----------+---------------+-----------+---------\n");
    }
    printf("(sink=%g)\n", (double)g_sink);
}

static void bench_one(const mrw_dispatch *d, uint32_t jc, uint32_t N, uint64_t target_instances, int best_of) {
    uint8_t *buf = NULL;
    size_t sz = build_rig_blob(jc, 8, MRW_CLIP_LOOPING, 4242u + jc, &buf);
    mrw_blob blob; mrw_blob_open(buf, sz, &blob);
    mrw_skeleton_view sv; mrw_blob_skeleton(&blob, &sv);
    mrw_clip_view cv;     mrw_clip_view_at(&blob, 1, &cv);

    mrw_mem_req sreq, oreq;
    mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F32, &sreq, &oreq);

    float  *times = (float *)malloc((size_t)N * sizeof(float));
    float dur = (float)(8 - 1) / RIG_FPS;
    for (uint32_t i = 0; i < N; ++i) times[i] = dur * ((float)i / (float)N); /* phase spread */

    uint8_t *outb = mrw_alloc64(oreq.size);
    uint8_t *scr  = mrw_alloc64(sreq.size);
    float *out = (float *)outb;
    float loopscr[RIG_MAX_JOINTS * 12];

    uint32_t reps = (uint32_t)(target_instances / N); if (reps < 1) reps = 1;

    /* warm-up */
    mrw_batch_clip_to_palette(d, &sv, &cv, times, N, out, oreq.size, scr, sreq.size);

    double best_batch = 1e30, best_loop = 1e30;
    for (int trial = 0; trial < best_of; ++trial) {
        double t0 = now_sec();
        for (uint32_t r = 0; r < reps; ++r)
            mrw_batch_clip_to_palette(d, &sv, &cv, times, N, out, oreq.size, scr, sreq.size);
        double t1 = now_sec();
        g_sink += out[0] + out[(size_t)(N - 1) * jc * 12u];
        if (t1 - t0 < best_batch) best_batch = t1 - t0;

        double t2 = now_sec();
        for (uint32_t r = 0; r < reps; ++r)
            for (uint32_t i = 0; i < N; ++i)
                mrw_clip_to_palette(&sv, &cv, times[i], loopscr,
                                    out + (size_t)i * jc * 12u, jc);
        double t3 = now_sec();
        g_sink += out[0];
        if (t3 - t2 < best_loop) best_loop = t3 - t2;
    }

    double insts = (double)reps * (double)N;
    double batch_ns = best_batch * 1e9 / insts;
    double loop_ns  = best_loop  * 1e9 / insts;
    printf("  %-8s | %3u | %8u | %10.2f | %10.2f | %12.3f | %6.2fx\n",
           backend_name(d), jc, N, batch_ns, loop_ns, batch_ns / jc, loop_ns / batch_ns);

    free(times); mrw_free(outb); mrw_free(scr); mrw_free(buf);
}

int main(int argc, char **argv) {
    int smoke = (argc > 1 && strcmp(argv[1], "--smoke") == 0);
    if (argc > 1 && strcmp(argv[1], "--isa") == 0) { bench_isa(); return 0; }

    mrw_dispatch dsc, dbest;
    mrw_dispatch_scalar(&dsc);
    mrw_dispatch_detect(&dbest);   /* best backend the host supports */
    int has_simd = (dbest.backend != MRW_BACKEND_SCALAR);

    printf("marrow batch benchmark - across-instance SoA (scalar vs SIMD) vs loop of single-instance AoS\n");
    printf("backend  | bones |        N | batch ns/i | loop ns/i  | batch ns/i/b | loop/batch\n");
    printf("---------+-------+----------+------------+------------+--------------+-----------\n");

    if (smoke) {
        bench_one(&dsc,   8,   8, 4096, 1);
        if (has_simd) bench_one(&dbest, 20, 256, 8192, 1);
        printf("(smoke ok)\n");
        return 0;
    }

    uint32_t bones[] = { 8, 12, 20 };
    uint32_t sizes[] = { 1, 64, 1024, 16384, 65536 };
    for (size_t b = 0; b < sizeof bones / sizeof bones[0]; ++b) {
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
            bench_one(&dsc, bones[b], sizes[s], 2000000u, 3);
            if (has_simd) bench_one(&dbest, bones[b], sizes[s], 2000000u, 3);
        }
        printf("---------+-------+----------+------------+------------+--------------+-----------\n");
    }
    printf("(sink=%g)\n", (double)g_sink);
    return 0;
}
