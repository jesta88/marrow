/* clock_gettime/CLOCK_MONOTONIC are POSIX names that glibc hides under strict -std=c11
 * (the demo target sets C_EXTENSIONS OFF). The feature macro must be defined before ANY
 * system header is pulled in, so it sits above every #include. Ignored by MSVC. */
#ifndef _WIN32
#  define _POSIX_C_SOURCE 199309L
#endif

#include "profiler.h"

#include <string.h>

/* ------------------------------------------------------------------ portable monotonic timer */
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
double prof_now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#  include <time.h>
double prof_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

/* ------------------------------------------------------------------ stat ring */

#define PROF_EMA_ALPHA 0.12   /* responsive but stable enough to read off the HUD */

static void stat_push(prof_stat *s, double ms) {
    s->samples[s->head] = ms;
    s->head = (s->head + 1u) % PROF_RING;
    if (s->count < PROF_RING) s->count++;
    s->ema = s->have_ema ? s->ema + PROF_EMA_ALPHA * (ms - s->ema) : ms;
    s->have_ema = 1;
}

double prof_mean(const prof_stat *s) {
    if (!s->count) return 0.0;
    double sum = 0.0;
    for (uint32_t i = 0; i < s->count; ++i) sum += s->samples[i];
    return sum / (double)s->count;
}

double prof_min(const prof_stat *s) {
    if (!s->count) return 0.0;
    double m = s->samples[0];
    for (uint32_t i = 1; i < s->count; ++i) if (s->samples[i] < m) m = s->samples[i];
    return m;
}

double prof_max(const prof_stat *s) {
    if (!s->count) return 0.0;
    double m = s->samples[0];
    for (uint32_t i = 1; i < s->count; ++i) if (s->samples[i] > m) m = s->samples[i];
    return m;
}

double prof_pct(const prof_stat *s, double pct) {
    if (!s->count) return 0.0;
    double tmp[PROF_RING];
    memcpy(tmp, s->samples, (size_t)s->count * sizeof tmp[0]);
    /* insertion sort - n<=128, called at HUD/summary cadence, not per draw */
    for (uint32_t i = 1; i < s->count; ++i) {
        double v = tmp[i]; uint32_t j = i;
        while (j > 0 && tmp[j - 1] > v) { tmp[j] = tmp[j - 1]; --j; }
        tmp[j] = v;
    }
    if (pct < 0.0) pct = 0.0; else if (pct > 1.0) pct = 1.0;
    uint32_t idx = (uint32_t)(pct * (double)(s->count - 1) + 0.5);
    return tmp[idx];
}

/* ------------------------------------------------------------------ lifecycle */

void prof_init(Profiler *p) {
    memset(p, 0, sizeof *p);
    for (int z = 0; z < PROF_CPU_COUNT; ++z) p->cpu_open[z] = -1.0;
    p->backend = "gpu";
    p->tier = "gpu-baked";
}

void prof_reset_stats(Profiler *p) {
    for (int z = 0; z < PROF_CPU_COUNT; ++z) memset(&p->cpu[z], 0, sizeof p->cpu[z]);
    for (int z = 0; z < PROF_GPU_COUNT; ++z) memset(&p->gpu[z], 0, sizeof p->gpu[z]);
    memset(&p->cpu_total, 0, sizeof p->cpu_total);
    memset(p->frame_ms, 0, sizeof p->frame_ms);
    p->frame_head = p->frame_count = 0;
    p->fps = 0.0;
    /* keep gpu_supported / counters / tier / backend; keep frame_mark + cpu_open/accum so an
     * in-progress frame's dt and any open span stay valid across the reset. */
}

static int is_wait(prof_cpu_zone z) {
    return z == PROF_WAIT_THROTTLE || z == PROF_ACQUIRE;
}

void prof_begin(Profiler *p, prof_cpu_zone z) { p->cpu_open[z] = prof_now_s(); }

void prof_end(Profiler *p, prof_cpu_zone z) {
    if (p->cpu_open[z] < 0.0) return;
    p->cpu_accum[z] += (prof_now_s() - p->cpu_open[z]) * 1000.0;
    p->cpu_open[z] = -1.0;
}

void prof_add_cpu(Profiler *p, prof_cpu_zone z, double ms) { p->cpu_accum[z] += ms; }

void prof_add_gpu(Profiler *p, prof_gpu_zone z, double ms) {
    p->gpu_supported = 1;
    stat_push(&p->gpu[z], ms);
}

void prof_frame_end(Profiler *p) {
    double work = 0.0;
    for (int z = 0; z < PROF_CPU_COUNT; ++z) {
        stat_push(&p->cpu[z], p->cpu_accum[z]);
        if (!is_wait((prof_cpu_zone)z)) work += p->cpu_accum[z];
    }
    stat_push(&p->cpu_total, work);
    p->cpu_frame_ms      = work;
    p->palette_gen_ms    = p->cpu_accum[PROF_PALETTE_GEN];
    p->palette_upload_ms = p->cpu_accum[PROF_PALETTE_UPLOAD];
    memset(p->cpu_accum, 0, sizeof p->cpu_accum);

    double now = prof_now_s();
    if (p->frame_mark > 0.0) {
        double dt_ms = (now - p->frame_mark) * 1000.0;
        p->frame_ms[p->frame_head] = (float)dt_ms;
        p->frame_head = (p->frame_head + 1u) % PROF_FRAMES;
        if (p->frame_count < PROF_FRAMES) p->frame_count++;
        double inst = dt_ms > 0.0 ? 1000.0 / dt_ms : 0.0;
        p->fps = p->fps > 0.0 ? p->fps + PROF_EMA_ALPHA * (inst - p->fps) : inst;
    }
    p->frame_mark = now;
}
