/* Measurement backbone for the demo - the one place timing lives.
 *
 * The profiler is built for *honest* numbers:
 *   - one portable monotonic timer (validate.c and the bench driver use prof_now_s, not private
 *     copies);
 *   - CPU scopes that SEPARATE waits (GPU backpressure / vsync) from work, so the headline CPU
 *     frame cost is update+palette+record+submit, not how long we blocked on the timeline;
 *   - PALETTE_GEN (the library batch kernel writing cacheable CPU scratch) timed SEPARATELY from
 *     PALETTE_UPLOAD (memcpy into write-combined mapped GPU memory), so "AVX2 < SSE2 < scalar"
 *     reflects the kernels rather than the memory system.
 *
 * Demo-only; the marrow runtime contains no timing code. Pure C11, no hot-path allocation. */
#ifndef DEMO_PROFILER_H
#define DEMO_PROFILER_H

#include <stdint.h>
#include <stddef.h>

/* Portable monotonic wall-clock seconds (Win32 QPC / POSIX CLOCK_MONOTONIC). */
double prof_now_s(void);

/* CPU timing scopes. WAITS come first and are excluded from the primary CPU frame metric. Keep
 * PROF_CPU_COUNT last. */
typedef enum {
    PROF_WAIT_THROTTLE = 0, /* timeline wait on this slot's prior submission (frame pacing) */
    PROF_ACQUIRE,           /* vkAcquireNextImageKHR                                         */
    PROF_CROWD_UPDATE,      /* baked GPU crowd time-advance + stage                          */
    PROF_HEROES_UPDATE,     /* CPU hero pose pipeline                                        */
    PROF_PALETTE_GEN,       /* mrw_batch_clip_to_palette into cacheable CPU scratch (library core) */
    PROF_PALETTE_UPLOAD,    /* memcpy scratch -> mapped GPU SSBO (write-combined)            */
    PROF_RECORD,            /* command-buffer recording                                      */
    PROF_SUBMIT,            /* vkc_end_frame submit + present                                */
    PROF_LOD,               /* unified field: distance partition + top-K + clip-pair compaction */
    PROF_CPU_COUNT
} prof_cpu_zone;

/* GPU timing zones - mirror vkc_gpu_zone (vk_context.h); fed from the timestamp readback. Keep
 * PROF_GPU_COUNT last and in the same order as vkc_gpu_zone. */
typedef enum {
    PROF_GPU_FRAME = 0,     /* whole-frame GPU time (begin_frame -> end_frame)               */
    PROF_GPU_GROUND,
    PROF_GPU_CROWD,
    PROF_GPU_HEROES,
    PROF_GPU_HUD,
    PROF_GPU_COUNT
} prof_gpu_zone;

#define PROF_RING   128   /* per-scope sample window for percentiles */
#define PROF_FRAMES 128   /* frame-time history ring for the HUD graph */

typedef struct {
    double   samples[PROF_RING];
    uint32_t head, count;
    double   ema;          /* exponential moving average (ms), stable HUD readout */
    int      have_ema;
} prof_stat;

typedef struct {
    /* CPU: one sample per zone per frame; spans within a frame accumulate (e.g. PALETTE_GEN is one
     * mrw_batch call per clip group, summed). */
    prof_stat cpu[PROF_CPU_COUNT];
    double    cpu_open[PROF_CPU_COUNT];   /* prof_now_s at an open span, <0 = none */
    double    cpu_accum[PROF_CPU_COUNT];  /* ms accumulated this frame              */

    /* GPU: one sample per zone per frame, fed from vk_context. */
    prof_stat gpu[PROF_GPU_COUNT];
    int       gpu_supported;              /* device exposes usable timestamps       */

    /* per-frame readouts (valid after prof_frame_end) */
    prof_stat cpu_total;                  /* sum of WORK scopes per frame (stable EMA for the HUD) */
    double    cpu_frame_ms;               /* sum of WORK scopes, waits excluded     */
    double    palette_gen_ms;             /* this frame's PALETTE_GEN               */
    double    palette_upload_ms;          /* this frame's PALETTE_UPLOAD            */

    /* whole-frame wall time history (incl. waits) for the graph + FPS */
    float     frame_ms[PROF_FRAMES];
    uint32_t  frame_head, frame_count;
    double    frame_mark;                 /* prof_now_s at the last frame boundary  */
    double    fps;                        /* smoothed                               */

    /* counters set by the render loop */
    uint64_t  instances, draws, triangles, bones;
    /* unified-field LOD split (field_r_a > 0 only in the field scene; 0 marks "not the field").
     * field_clamped > 0 means more than near_cap entities fell inside R_A, so the surplus rendered
     * Tier B this frame (the bounded, honest near-cap fallback). */
    uint32_t  field_near, field_far, field_clamped;
    float     field_r_a;
    uint64_t  gen_bones;                  /* bone-instances the PALETTE_GEN kernel covered this frame
                                             (cpu-crowd count x joints); 0 on the GPU tier. Lets the HUD
                                             report scale-invariant ns/(instance.bone) without the
                                             heroes' bones skewing it at low crowd counts. */
    const char *backend;                  /* "scalar" / "SSE2" / "AVX2" / "gpu"     */
    const char *tier;                     /* "gpu-baked" / "cpu-batch"              */
    const char *model;                    /* active model name (M cycles)           */
} Profiler;

void prof_init(Profiler *p);
/* Clear the rolling stats (per-scope rings + EMA, frame-time history, FPS) WITHOUT disturbing the
 * device-capability flag, counters, tier/backend labels, or the frame clock. Call after a config
 * change (count / backend) so steady-state percentiles settle without carrying the transition. */
void prof_reset_stats(Profiler *p);

/* Scope timing - begin/end bracket a CPU span; spans for the same zone accumulate within a frame. */
void prof_begin(Profiler *p, prof_cpu_zone z);
void prof_end  (Profiler *p, prof_cpu_zone z);
/* Add a pre-measured CPU span (ms) without bracketing (e.g. the bench driver times inline). */
void prof_add_cpu(Profiler *p, prof_cpu_zone z, double ms);
/* Feed a GPU zone result (ms) from the vk_context timestamp readback. */
void prof_add_gpu(Profiler *p, prof_gpu_zone z, double ms);

/* Close out the frame: roll each CPU accumulator into its ring, roll the frame-time ring + FPS,
 * publish the per-frame readouts, and reset the accumulators. Call once, after every scope closes. */
void prof_frame_end(Profiler *p);

double prof_mean(const prof_stat *s);
double prof_min (const prof_stat *s);
double prof_max (const prof_stat *s);
double prof_pct (const prof_stat *s, double pct);  /* percentile, pct in [0,1]; 0 if empty */

#endif /* DEMO_PROFILER_H */
