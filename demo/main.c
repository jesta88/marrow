/* marrow Vulkan demo - entry point.
 *
 * Showcases the marrow animation runtime across both tiers and MEASURES it honestly:
 *   - baked GPU crowd (crowd.c) - animation sampled from the baked palette on the GPU;
 *   - CPU heroes (heroes.c) with LOD promotion (P) to the baked path;
 *   - a live CPU-batch crowd tier (crowd_cpu.c) driving the flagship mrw_batch_clip_to_palette
 *     kernel every frame on a runtime-selectable backend;
 *   - an in-window HUD (F1) with CPU/GPU per-stage timings, and a headless --bench sweep.
 *
 * Controls: WASD/Q/E fly, hold right-mouse to look, Shift sprint, Esc quit.
 *           M cycle model (procedural biped + every baked model in the assets folder),
 *           P promote heroes (CPU <-> baked GPU), T toggle crowd tier (GPU <-> CPU),
 *           B cycle CPU backend (scalar/SSE2/AVX2), 1/2/3 pick backend directly,
 *           J toggle multithreaded CPU-batch (palette-gen across cores),
 *           H toggle CPU-batch palette f32 <-> f16 (half the write/upload/fetch),
 *           -/= halve/double the live entity count, R reset perf stats,
 *           F1 toggle HUD, F2 HUD detail.
 * Flags: --count N, --promote, --crowd-tier {gpu|cpu}, --crowd-f16 (CPU tier, f16 palette),
 *        --frames N, --screenshot PATH, --validate, --validate-gpu,
 *        --bench [--bench-out FILE] [--smoke], --selftest-assets, --cycle-models,
 *        --showcase (CPU additive/mask/aim/two-bone-IK scene on the biped),
 *        --gltf PATH [--mrw PATH]. */
#include <volk.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vk_context.h"
#include "camera.h"
#include "linalg.h"
#include "assets_proc.h"
#include "assets_gltf.h"
#include "models.h"
#include "crowd.h"
#include "crowd_cpu.h"
#include "heroes.h"
#include "hud.h"
#include "jobs.h"
#include "profiler.h"
#include "validate.h"

#include "ground_vert.spv.h"
#include "ground_frag.spv.h"

#ifdef NDEBUG
#  define BUILD_STR "Release"
#else
#  define BUILD_STR "Debug"
#endif

/* ------------------------------------------------------------------ ground */

static const float g_ground_verts[] = {
    -120.0f, 0.0f, -120.0f,  120.0f, 0.0f, -120.0f,  120.0f, 0.0f, 120.0f,  -120.0f, 0.0f, 120.0f,
};
static const uint16_t g_ground_idx[] = { 0, 2, 1, 0, 3, 2 };

typedef struct {
    VkShaderEXT vs, fs;
    VkPipelineLayout layout;
    VkBuffer vbuf, ibuf; VkDeviceMemory vmem, imem;
} Ground;

static int ground_init(Ground *g, VkCtx *ctx) {
    memset(g, 0, sizeof *g);
    VkPushConstantRange pr = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mat4) };
    VkPipelineLayoutCreateInfo li = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(ctx->device, &li, NULL, &g->layout) != VK_SUCCESS) return 0;
    if (!vkc_create_graphics_shaders(ctx, ground_vert_spv, sizeof ground_vert_spv,
            ground_frag_spv, sizeof ground_frag_spv, NULL, 0, &pr, &g->vs, &g->fs)) return 0;
    void *p;
    if (!vkc_create_host_buffer(ctx, sizeof g_ground_verts, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &g->vbuf, &g->vmem, &p)) return 0;
    memcpy(p, g_ground_verts, sizeof g_ground_verts);
    if (!vkc_create_host_buffer(ctx, sizeof g_ground_idx, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &g->ibuf, &g->imem, &p)) return 0;
    memcpy(p, g_ground_idx, sizeof g_ground_idx);
    return 1;
}

static void ground_draw(Ground *g, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *vp, VkExtent2D extent) {
    VkVertexInputBindingDescription2EXT b = { VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT };
    b.binding = 0; b.stride = 3 * sizeof(float); b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; b.divisor = 1;
    VkVertexInputAttributeDescription2EXT a = { VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT };
    a.location = 0; a.binding = 0; a.format = VK_FORMAT_R32G32B32_SFLOAT; a.offset = 0;
    vkc_bind_shaders(ctx, cmd, g->vs, g->fs);
    vkc_set_default_state(ctx, cmd, extent, &b, 1, &a, 1);
    vkCmdPushConstants(cmd, g->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mat4), vp->m);
    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &g->vbuf, &zero);
    vkCmdBindIndexBuffer(cmd, g->ibuf, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
}

static void ground_destroy(Ground *g, VkCtx *ctx) {
    vkc_destroy_shader(ctx, g->vs); vkc_destroy_shader(ctx, g->fs);
    if (g->layout) vkDestroyPipelineLayout(ctx->device, g->layout, NULL);
    vkc_destroy_buffer(ctx, g->vbuf, g->vmem);
    vkc_destroy_buffer(ctx, g->ibuf, g->imem);
}

/* ------------------------------------------------------------------ scene (shared by interactive + bench) */

typedef struct {
    VkCtx       *ctx;
    Profiler    *prof;
    const SharedMesh *mesh;
    Ground      *ground;
    Crowd       *crowd;   /* NULL when the rig has no BAKED section (CPU-only) */
    CrowdCpu    *ccpu;
    Heroes      *heroes;  /* NULL in --bench */
    Hud         *hud;     /* NULL in --bench */
    uint32_t     joint_count;
    const char  *model;   /* active model name, for the HUD/title */
    CrowdDrawMode crowd_mode;   /* FULL interactively; the bench sweeps the discard variants */
} Scene;

/* Simulate + record + submit one frame. tier: 0 = GPU-baked crowd, 1 = CPU-batch crowd. Returns 0
 * if the frame was skipped (swapchain rebuild). dt is injected (real time interactively, fixed in
 * --bench), so animation advances reproducibly. */
static int scene_frame(Scene *s, Camera *cam, float dt, int tier,
                       int promote, int draw_heroes, int draw_hud) {
    VkCtx *ctx = s->ctx; Profiler *prof = s->prof;
    VkCommandBuffer cmd; VkExtent2D extent;
    if (!vkc_begin_frame(ctx, &cmd, &extent)) return 0;

    /* waits (measured in begin_frame) are kept SEPARATE from the work scopes */
    prof_add_cpu(prof, PROF_WAIT_THROTTLE, ctx->wait_ms);
    prof_add_cpu(prof, PROF_ACQUIRE, ctx->acquire_ms);
    float gms[VKC_GPU_ZONE_COUNT];
    uint32_t gmask = vkc_gpu_results_ms(ctx, gms);
    for (uint32_t z = 0; z < VKC_GPU_ZONE_COUNT; ++z)
        if (gmask & (1u << z)) prof_add_gpu(prof, (prof_gpu_zone)z, gms[z]);

    int use_gpu_crowd = (tier == 0) && (s->crowd != NULL);
    if (use_gpu_crowd) {
        prof_begin(prof, PROF_CROWD_UPDATE);
        crowd_update(s->crowd, dt, ctx->cur_frame);
        prof_end(prof, PROF_CROWD_UPDATE);
    } else {
        crowd_cpu_update(s->ccpu, dt, ctx->cur_frame, prof);  /* times CROWD_UPDATE/GEN/UPLOAD */
    }
    if (draw_heroes) {
        prof_begin(prof, PROF_HEROES_UPDATE);
        heroes_update(s->heroes, dt, ctx->cur_frame);
        prof_end(prof, PROF_HEROES_UPDATE);
    }

    float aspect = (float)extent.width / (float)extent.height;
    mat4 view_proj = camera_view_proj(cam, aspect);

    /* Publish counters BEFORE recording so the HUD (drawn during PROF_RECORD) reads THIS frame's
     * tier / backend / instance / gen_bones values rather than the previous frame's - matters the
     * frame a tier or count change lands. */
    uint64_t crowd_count = use_gpu_crowd ? s->crowd->count : s->ccpu->count;
    uint64_t inst = crowd_count + (draw_heroes ? s->heroes->count : 0);
    uint64_t tris_per = s->mesh->index_count / 3;
    prof->instances = inst;
    prof->triangles = tris_per * inst + 2;     /* + ground quad */
    prof->bones     = (uint64_t)s->joint_count * inst;
    prof->gen_bones = use_gpu_crowd ? 0 : (uint64_t)s->joint_count * s->ccpu->count;  /* PALETTE_GEN coverage */
    prof->draws     = 2 + (draw_heroes ? 1u : 0u) + (draw_hud ? 1u : 0u);
    prof->tier      = use_gpu_crowd ? "gpu-baked" : (s->ccpu->use_f16 ? "cpu-batch-f16" : "cpu-batch");
    prof->backend   = use_gpu_crowd ? "gpu" : s->ccpu->backend_name;
    prof->model     = s->model;

    prof_begin(prof, PROF_RECORD);
    vkc_gpu_zone_begin(ctx, cmd, VKC_GPU_ZONE_GROUND);
    ground_draw(s->ground, ctx, cmd, &view_proj, extent);
    vkc_gpu_zone_end(ctx, cmd, VKC_GPU_ZONE_GROUND);

    vkc_gpu_zone_begin(ctx, cmd, VKC_GPU_ZONE_CROWD);
    if (use_gpu_crowd) crowd_draw(s->crowd, ctx, cmd, &view_proj, extent, s->crowd_mode);
    else               crowd_cpu_draw(s->ccpu, ctx, cmd, &view_proj, extent);
    vkc_gpu_zone_end(ctx, cmd, VKC_GPU_ZONE_CROWD);

    if (draw_heroes) {
        vkc_gpu_zone_begin(ctx, cmd, VKC_GPU_ZONE_HEROES);
        if (promote && s->crowd)
            crowd_draw_instances(s->crowd, ctx, cmd, &view_proj, extent,
                                 s->heroes->tierb_addr[ctx->cur_frame], s->heroes->count, CROWD_DRAW_FULL);
        else
            heroes_draw(s->heroes, ctx, cmd, &view_proj, extent,
                        s->mesh->vbuf, s->mesh->ibuf, s->mesh->index_count);
        vkc_gpu_zone_end(ctx, cmd, VKC_GPU_ZONE_HEROES);
    }

    if (draw_hud) hud_draw(s->hud, ctx, cmd, extent, prof);   /* brackets its own HUD GPU zone */
    prof_end(prof, PROF_RECORD);

    prof_begin(prof, PROF_SUBMIT);
    vkc_end_frame(ctx);
    prof_end(prof, PROF_SUBMIT);

    prof_frame_end(prof);
    return 1;
}

/* ------------------------------------------------------------------ --bench */

typedef struct {
    char     tier[16], backend[12];
    uint32_t count;
    double   cpu_mean, cpu_p95, gpu_mean, gpu_p95, gpu_crowd, gen, upload, fps;
    /* GPU vertex-skinning microbenchmark (rasterizer-discard crowd zone), gpu-baked tier only:
     * gpu_skin = real skinning VS, gpu_static = matched no-palette baseline; the difference isolates
     * the marrow palette sample+decode+blend. has_skin gates whether they were measured. */
    double   gpu_skin, gpu_static;
    int      gpu_ok, has_skin;
} BenchRow;

static void drive_frame(Scene *s, Camera *cam, float dt, int tier) {
    for (int tries = 0; tries < 16; ++tries)
        if (scene_frame(s, cam, dt, tier, 0, 0, 0)) return;   /* no heroes / no HUD in bench */
}

/* Warm then measure the GPU crowd-zone time (ms) for one crowd draw mode, in its OWN profiler so
 * full/skin/static are independently warmed (no back-to-back cache bias). Returns -1 if untimed. */
static double measure_gpu_crowd_ms(Scene *s, Camera *cam, CrowdDrawMode mode, int warm, int meas) {
    Profiler prof; prof_init(&prof);
    s->prof = &prof;
    CrowdDrawMode saved = s->crowd_mode;
    s->crowd_mode = mode;
    const float dt = 1.0f / 60.0f;
    for (int i = 0; i < warm; ++i) drive_frame(s, cam, dt, 0);   /* tier 0 = gpu-baked */
    prof_init(&prof);
    for (int i = 0; i < meas; ++i) drive_frame(s, cam, dt, 0);
    s->crowd_mode = saved;
    return prof.gpu_supported ? prof_mean(&prof.gpu[PROF_GPU_CROWD]) : -1.0;
}

static void run_config(Scene *s, Camera *cam, int tier, int warm, int meas, BenchRow *row) {
    Profiler prof; prof_init(&prof);
    s->prof = &prof;
    const float dt = 1.0f / 60.0f;
    for (int i = 0; i < warm; ++i) drive_frame(s, cam, dt, tier);
    prof_init(&prof);                       /* reset stats for the measurement window */
    for (int i = 0; i < meas; ++i) drive_frame(s, cam, dt, tier);

    row->cpu_mean = prof_mean(&prof.cpu_total);
    row->cpu_p95  = prof_pct(&prof.cpu_total, 0.95);
    row->gpu_ok   = prof.gpu_supported;
    row->gpu_mean = prof.gpu_supported ? prof_mean(&prof.gpu[PROF_GPU_FRAME]) : -1.0;
    row->gpu_p95  = prof.gpu_supported ? prof_pct(&prof.gpu[PROF_GPU_FRAME], 0.95) : -1.0;
    row->gpu_crowd = prof.gpu_supported ? prof_mean(&prof.gpu[PROF_GPU_CROWD]) : -1.0;
    row->gen      = prof_mean(&prof.cpu[PROF_PALETTE_GEN]);
    row->upload   = prof_mean(&prof.cpu[PROF_PALETTE_UPLOAD]);
    row->fps      = prof.fps;
}

static const char *has_suffix(const char *s, const char *suf) {
    size_t n = strlen(s), m = strlen(suf);
    return (n >= m && strcmp(s + n - m, suf) == 0) ? s + n - m : NULL;
}

/* Isolated marrow baked-GPU palette skinning, ns per (instance·vertex): (skin-static) over the
 * vertices the crowd draw shaded. Per-VERTEX (not per-bone): the GPU re-skins per vertex·influence,
 * unlike the CPU palette which is per (instance·bone), so the two denominators differ. */
static double skin_ns_per_vert(const BenchRow *r, uint32_t vert_count) {
    if (!r->has_skin || vert_count == 0 || r->count == 0) return -1.0;
    return (r->gpu_skin - r->gpu_static) * 1e6 / ((double)r->count * (double)vert_count);
}

static void bench_print_table(FILE *f, const char *dev, uint32_t vbits, uint32_t vert_count,
                              const BenchRow *r, int n) {
    fprintf(f, "\n=== marrow demo benchmark (%s) ===\n", BUILD_STR);
    fprintf(f, "device: %s  (timestamp valid bits %u, mesh %u verts/instance)\n", dev, vbits, vert_count);
    fprintf(f, "%-10s %-7s %8s | %8s %8s | %8s %8s %8s | %8s %8s %9s | %8s %8s | %8s\n",
            "tier", "backend", "count", "cpu_mean", "cpu_p95", "gpu_mean", "gpu_p95", "gpu_crwd",
            "gpu_skin", "gpu_stat", "skin_ns/v", "gen_ms", "upl_ms", "fps");
    for (int i = 0; i < n; ++i) {
        char skin[16], stat[16], nsv[16];
        if (r[i].has_skin) {
            double d = skin_ns_per_vert(&r[i], vert_count);
            snprintf(skin, sizeof skin, "%8.3f", r[i].gpu_skin);
            snprintf(stat, sizeof stat, "%8.3f", r[i].gpu_static);
            snprintf(nsv,  sizeof nsv,  "%9.3f", d);
        } else {
            snprintf(skin, sizeof skin, "%8s", "n/a");
            snprintf(stat, sizeof stat, "%8s", "n/a");
            snprintf(nsv,  sizeof nsv,  "%9s", "n/a");
        }
        if (r[i].gpu_ok)
            fprintf(f, "%-10s %-7s %8u | %8.3f %8.3f | %8.3f %8.3f %8.3f | %s %s %s | %8.4f %8.4f | %8.1f\n",
                    r[i].tier, r[i].backend, r[i].count, r[i].cpu_mean, r[i].cpu_p95,
                    r[i].gpu_mean, r[i].gpu_p95, r[i].gpu_crowd, skin, stat, nsv,
                    r[i].gen, r[i].upload, r[i].fps);
        else
            fprintf(f, "%-10s %-7s %8u | %8.3f %8.3f | %8s %8s %8s | %s %s %s | %8.4f %8.4f | %8.1f\n",
                    r[i].tier, r[i].backend, r[i].count, r[i].cpu_mean, r[i].cpu_p95,
                    "n/a", "n/a", "n/a", skin, stat, nsv, r[i].gen, r[i].upload, r[i].fps);
    }
}

static int bench_write_csv(const char *path, const char *dev, uint32_t vbits, int gpu_ok,
                           const BenchRow *r, int n) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "[bench] cannot open %s\n", path); return 1; }
    fprintf(f, "# marrow demo bench  device=%s  timestamp_valid_bits=%u  build=%s  gpu_timestamps=%d\n",
            dev, vbits, BUILD_STR, gpu_ok);
    fprintf(f, "tier,backend,count,cpu_ms_mean,cpu_ms_p95,gpu_ms_mean,gpu_ms_p95,gpu_ms_crowd,"
               "gpu_ms_skin,gpu_ms_static,palette_gen_ms,palette_upload_ms,fps\n");
    for (int i = 0; i < n; ++i) {
        char skin[16], stat[16];
        if (r[i].has_skin) { snprintf(skin, sizeof skin, "%.4f", r[i].gpu_skin);
                             snprintf(stat, sizeof stat, "%.4f", r[i].gpu_static); }
        else               { snprintf(skin, sizeof skin, "n/a"); snprintf(stat, sizeof stat, "n/a"); }
        if (r[i].gpu_ok)
            fprintf(f, "%s,%s,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,%.4f,%.4f,%.1f\n",
                    r[i].tier, r[i].backend, r[i].count, r[i].cpu_mean, r[i].cpu_p95,
                    r[i].gpu_mean, r[i].gpu_p95, r[i].gpu_crowd, skin, stat, r[i].gen, r[i].upload, r[i].fps);
        else
            fprintf(f, "%s,%s,%u,%.4f,%.4f,n/a,n/a,n/a,%s,%s,%.4f,%.4f,%.1f\n",
                    r[i].tier, r[i].backend, r[i].count, r[i].cpu_mean, r[i].cpu_p95,
                    skin, stat, r[i].gen, r[i].upload, r[i].fps);
    }
    fclose(f);
    return 0;
}

static int bench_write_json(const char *path, const char *dev, uint32_t vbits, int gpu_ok,
                            const BenchRow *r, int n) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "[bench] cannot open %s\n", path); return 1; }
    fprintf(f, "{\n  \"device\": \"%s\",\n  \"timestamp_valid_bits\": %u,\n", dev, vbits);
    fprintf(f, "  \"build\": \"%s\",\n  \"gpu_timestamps\": %s,\n  \"results\": [\n",
            BUILD_STR, gpu_ok ? "true" : "false");
    for (int i = 0; i < n; ++i) {
        fprintf(f, "    {\"tier\":\"%s\",\"backend\":\"%s\",\"count\":%u,"
                   "\"cpu_ms_mean\":%.4f,\"cpu_ms_p95\":%.4f,",
                r[i].tier, r[i].backend, r[i].count, r[i].cpu_mean, r[i].cpu_p95);
        if (r[i].gpu_ok) fprintf(f, "\"gpu_ms_mean\":%.4f,\"gpu_ms_p95\":%.4f,\"gpu_ms_crowd\":%.4f,",
                                 r[i].gpu_mean, r[i].gpu_p95, r[i].gpu_crowd);
        else             fprintf(f, "\"gpu_ms_mean\":null,\"gpu_ms_p95\":null,\"gpu_ms_crowd\":null,");
        if (r[i].has_skin) fprintf(f, "\"gpu_ms_skin\":%.4f,\"gpu_ms_static\":%.4f,",
                                   r[i].gpu_skin, r[i].gpu_static);
        else               fprintf(f, "\"gpu_ms_skin\":null,\"gpu_ms_static\":null,");
        fprintf(f, "\"palette_gen_ms\":%.4f,\"palette_upload_ms\":%.4f,\"fps\":%.1f}%s\n",
                r[i].gen, r[i].upload, r[i].fps, i + 1 < n ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    return 0;
}

/* In-memory schema self-check. Beyond per-row sanity (finite fields, count>0, frames actually
 * timed), it proves the sweep COVERED what this host can run: the want_* flags are the same
 * availability bits the sweep itself branched on, so a silently-dropped GPU-baked path, missing
 * SSE2/AVX2 backend, or dead GPU-timestamp readback fails the gate instead of passing with fewer
 * rows. Catches a malformed sweep at the source (the CTest also re-parses the emitted files). */
static int bench_self_check(const BenchRow *r, int n, int want_baked, int want_sse2,
                            int want_avx2, int want_gpu_ts) {
    if (n < 1) { fprintf(stderr, "[bench] no rows produced\n"); return 1; }
    int have_cpu_scalar = 0, have_baked = 0, have_sse2 = 0, have_avx2 = 0;
    for (int i = 0; i < n; ++i) {
        if (r[i].count == 0 || !isfinite(r[i].cpu_mean) || !isfinite(r[i].cpu_p95) ||
            !isfinite(r[i].gen) || !isfinite(r[i].upload) || !isfinite(r[i].fps)) {
            fprintf(stderr, "[bench] row %d has a non-finite field\n", i); return 1;
        }
        if (!(r[i].fps > 0.0)) {   /* no frames were actually timed -> a broken/under-sampled run */
            fprintf(stderr, "[bench] row %d measured no frames (fps=0)\n", i); return 1;
        }
        if (r[i].gpu_ok && (!isfinite(r[i].gpu_mean) || !isfinite(r[i].gpu_p95))) {
            fprintf(stderr, "[bench] row %d gpu field non-finite\n", i); return 1;
        }
        /* If the device exposes timestamps, every rendered row must carry GPU timing (each frame
         * stamps at least the whole-frame zone); a row missing it means a dead readback path. */
        if (want_gpu_ts && !r[i].gpu_ok) {
            fprintf(stderr, "[bench] row %d has no GPU timestamps but the device supports them\n", i); return 1;
        }
        int is_baked = strcmp(r[i].tier, "gpu-baked") == 0;
        /* The gpu-baked tier runs the vertex-skinning microbenchmark; with timestamps on it must
         * have produced finite, non-negative skin/static numbers (a missing pair = a dead bench mode). */
        if (is_baked && want_gpu_ts) {
            if (!r[i].has_skin) { fprintf(stderr, "[bench] gpu-baked row %d missing skin/static measurement\n", i); return 1; }
            if (!isfinite(r[i].gpu_skin) || !isfinite(r[i].gpu_static) ||
                r[i].gpu_skin < 0.0 || r[i].gpu_static < 0.0) {
                fprintf(stderr, "[bench] gpu-baked row %d skin/static non-finite or negative\n", i); return 1;
            }
            /* At scale, the isolated palette work (skin-static) must be non-negative: if the skinning
             * were dead-code-eliminated under rasterizer discard, skin would collapse toward static. */
            if (r[i].count >= 4096u && r[i].gpu_skin < r[i].gpu_static) {
                fprintf(stderr, "[bench] gpu-baked row %d: gpu_skin (%.4f) < gpu_static (%.4f) at count %u "
                        "- skinning may have been optimized away under rasterizer discard\n",
                        i, r[i].gpu_skin, r[i].gpu_static, r[i].count); return 1;
            }
        }
        int is_cpu = strcmp(r[i].tier, "cpu-batch") == 0;
        if (is_cpu && strcmp(r[i].backend, "scalar") == 0) have_cpu_scalar = 1;
        if (is_cpu && strcmp(r[i].backend, "SSE2")   == 0) have_sse2 = 1;
        if (is_cpu && strcmp(r[i].backend, "AVX2")   == 0) have_avx2 = 1;
        if (strcmp(r[i].tier, "gpu-baked") == 0)           have_baked = 1;
    }
    if (!have_cpu_scalar)          { fprintf(stderr, "[bench] missing cpu-batch/scalar config\n"); return 1; }
    if (want_sse2  && !have_sse2)  { fprintf(stderr, "[bench] host supports SSE2 but produced no SSE2 row\n"); return 1; }
    if (want_avx2  && !have_avx2)  { fprintf(stderr, "[bench] host supports AVX2 but produced no AVX2 row\n"); return 1; }
    if (want_baked && !have_baked) { fprintf(stderr, "[bench] Tier-B eligible but produced no gpu-baked row\n"); return 1; }
    return 0;
}

static int bench_run(VkCtx *ctx, const ProcAssets *assets, const SharedMesh *mesh,
                     Ground *ground, int smoke, const char *out_path) {
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(ctx->phys, &props);

    mrw_dispatch tmp;
    int has_sse2 = (mrw_dispatch_sse2(&tmp) == MRW_OK);
    int has_avx2 = (mrw_dispatch_avx2(&tmp) == MRW_OK);
    int has_baked = assets->tier_b_eligible;

    const uint32_t full_counts[]  = { 1000u, 4000u, 16384u, 65536u };
    const uint32_t smoke_counts[] = { 256u };
    const uint32_t *counts = smoke ? smoke_counts : full_counts;
    int ncount = smoke ? 1 : (int)(sizeof full_counts / sizeof full_counts[0]);
    int warm = smoke ? 4 : 30, meas = smoke ? 8 : 120;

    Camera cam; camera_init(&cam, v3(0.0f, 28.0f, 26.0f), 0.0f, -0.5f);

    BenchRow rows[64]; int nrows = 0;
    fprintf(stderr, "[bench] %s sweep, device %s\n", smoke ? "smoke" : "full", props.deviceName);

    for (int ci = 0; ci < ncount; ++ci) {
        uint32_t count = counts[ci];
        Crowd crowd; int have_crowd = 0;
        CrowdCpu ccpu; int have_ccpu = 0;
        Scene s; memset(&s, 0, sizeof s);
        s.ctx = ctx; s.mesh = mesh; s.ground = ground; s.joint_count = assets->joint_count;
        s.model = "(bench)";

        if (has_baked && crowd_init(&crowd, ctx, assets, mesh, count, count)) { have_crowd = 1; s.crowd = &crowd; }
        if (count <= CROWD_CPU_MAX && crowd_cpu_init(&ccpu, ctx, assets, mesh, count, count)) { have_ccpu = 1; s.ccpu = &ccpu; }

        if (have_crowd && nrows < 64) {
            BenchRow *r = &rows[nrows++];
            memset(r, 0, sizeof *r);
            s.crowd_mode = CROWD_DRAW_FULL;
            run_config(&s, &cam, 0, warm, meas, r);   /* full render (headline frame cost) */
            snprintf(r->tier, sizeof r->tier, "gpu-baked");
            snprintf(r->backend, sizeof r->backend, "gpu");
            r->count = count;
            /* GPU vertex-skinning microbenchmark: full/skin/static are each independently warmed
             * (separate profilers) so the subtraction isn't biased by back-to-back cache state. */
            if (r->gpu_ok) {
                r->gpu_skin   = measure_gpu_crowd_ms(&s, &cam, CROWD_DRAW_SKIN_DISCARD, warm, meas);
                r->gpu_static = measure_gpu_crowd_ms(&s, &cam, CROWD_DRAW_STATIC_DISCARD, warm, meas);
                r->has_skin   = (r->gpu_skin >= 0.0 && r->gpu_static >= 0.0);
            }
        }
        if (have_ccpu) {
            mrw_backend bks[3]; int nbk = 0;
            bks[nbk++] = MRW_BACKEND_SCALAR;
            if (has_sse2) bks[nbk++] = MRW_BACKEND_SSE2;
            if (has_avx2) bks[nbk++] = MRW_BACKEND_AVX2;
            for (int b = 0; b < nbk && nrows < 64; ++b) {
                crowd_cpu_set_backend(&ccpu, bks[b]);
                BenchRow *r = &rows[nrows++];
                memset(r, 0, sizeof *r);
                run_config(&s, &cam, 1, warm, meas, r);
                snprintf(r->tier, sizeof r->tier, "cpu-batch");
                snprintf(r->backend, sizeof r->backend, "%s", ccpu.backend_name);
                r->count = count;
            }
        }
        if (have_ccpu) crowd_cpu_destroy(&ccpu, ctx);
        if (have_crowd) crowd_destroy(&crowd, ctx);
    }

    bench_print_table(stdout, props.deviceName, ctx->timestamp_valid_bits, mesh->vert_count, rows, nrows);

    /* Shared pure-CPU kernel microbench: the isolated scalar/SSE2/AVX2 comparison (the live in-loop
     * PALETTE_GEN column above is the per-frame number). Same helper --validate uses. */
    {
        mrw_blob blob;
        if (mrw_blob_open(assets->blob, assets->blob_size, &blob) == MRW_OK && assets->clip_count > 0) {
            mrw_skeleton_view skel; mrw_blob_skeleton(&blob, &skel);
            mrw_clip_view clip;
            if (mrw_clip_view_at(&blob, 1 + assets->clips[0].clip_index, &clip) == MRW_OK) {
                fprintf(stderr, "\n");
                validate_microbench(&skel, &clip, assets->clips[0].duration_s, smoke ? 1024u : 16384u, stderr);
            }
        }
    }

    int rc = 0;
    if (out_path) {
        if (has_suffix(out_path, ".json"))
            rc |= bench_write_json(out_path, props.deviceName, ctx->timestamp_valid_bits,
                                   ctx->timestamp_valid_bits != 0, rows, nrows);
        else if (has_suffix(out_path, ".csv"))
            rc |= bench_write_csv(out_path, props.deviceName, ctx->timestamp_valid_bits,
                                  ctx->timestamp_valid_bits != 0, rows, nrows);
        else
            fprintf(stderr, "[bench] --bench-out needs a .csv or .json extension; skipping file\n");
    }
    rc |= bench_self_check(rows, nrows, has_baked, has_sse2, has_avx2, ctx->timestamp_valid_bits != 0);
    return rc;
}

/* ------------------------------------------------------------------ main */

/* Geometric step for the live entity count (-/= keys): doubling/halving sweeps orders of magnitude
 * in a few keypresses, which is what you want when watching CPU/GPU cost scale. Clamped to [1, cap]. */
static uint32_t count_step(uint32_t c, int up, uint32_t cap) {
    uint32_t n = up ? (c > cap / 2u ? cap : c * 2u) : (c <= 1u ? 1u : c / 2u);   /* no *2 overflow */
    if (n > cap) n = cap;
    if (n < 1u) n = 1u;
    return n;
}

/* ------------------------------------------------------------------ per-model GPU resources
 * Everything that depends on the loaded character - the .mrw assets, the shared skinned mesh, and
 * the three skinning paths (baked GPU crowd, CPU-batch crowd, CPU heroes). Bundled so the M-key
 * model switch can tear it all down and rebuild it for the next model while the device-, window-,
 * ground-, HUD-, and job-pool-level state lives on. */
#define DEMO_HERO_COUNT 14u

typedef struct {
    ProcAssets assets;
    SharedMesh mesh;
    Crowd      crowd; int have_crowd;   /* crowd absent when the rig has no BAKED section */
    CrowdCpu   ccpu;
    Heroes     heroes;
} ModelGpu;

static int load_assets(const ModelSource *src, ProcAssets *out) {
    if (src->kind == MODEL_PROCEDURAL) return assets_proc_build(out);
    return assets_gltf_build(src->mrw, src->gltf, out);
}

static void model_gpu_destroy(ModelGpu *m, VkCtx *ctx) {
    /* heroes + ccpu hold marrow views that borrow assets->blob, so free assets last. */
    heroes_destroy(&m->heroes, ctx);
    crowd_cpu_destroy(&m->ccpu, ctx);
    if (m->have_crowd) crowd_destroy(&m->crowd, ctx);
    shared_mesh_destroy(&m->mesh, ctx);
    assets_proc_free(&m->assets);
    memset(m, 0, sizeof *m);
}

/* Build all model-dependent resources for `src`. Returns 0 with nothing leaked on failure (any
 * partially-built state is torn down). `pool` is borrowed; `crowd_f16` selects the CPU tier format. */
static int model_gpu_load(ModelGpu *m, VkCtx *ctx, const ModelSource *src,
                          uint32_t capacity, uint32_t count, int crowd_f16, Jobs *pool) {
    memset(m, 0, sizeof *m);
    if (load_assets(src, &m->assets) != 0) {
        fprintf(stderr, "[demo] asset load failed for '%s'\n", src->name); return 0;
    }
    if (!shared_mesh_init(&m->mesh, ctx, &m->assets)) {
        fprintf(stderr, "[demo] shared mesh init failed\n");
        assets_proc_free(&m->assets); return 0;
    }
    if (m->assets.tier_b_eligible && crowd_init(&m->crowd, ctx, &m->assets, &m->mesh, capacity, count))
        m->have_crowd = 1;
    if (!crowd_cpu_init(&m->ccpu, ctx, &m->assets, &m->mesh, capacity, count)) {
        fprintf(stderr, "[demo] cpu crowd init failed\n");
        if (m->have_crowd) crowd_destroy(&m->crowd, ctx);
        shared_mesh_destroy(&m->mesh, ctx); assets_proc_free(&m->assets); return 0;
    }
    if (crowd_f16) crowd_cpu_set_f16(&m->ccpu, 1);
    if (pool) crowd_cpu_set_jobs(&m->ccpu, pool);
    if (!heroes_init(&m->heroes, ctx, &m->assets, DEMO_HERO_COUNT)) {
        fprintf(stderr, "[demo] heroes init failed\n");
        crowd_cpu_destroy(&m->ccpu, ctx);
        if (m->have_crowd) crowd_destroy(&m->crowd, ctx);
        shared_mesh_destroy(&m->mesh, ctx); assets_proc_free(&m->assets); return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    int max_frames = 0; const char *shot = NULL; int selftest_assets = 0; uint32_t count = 1600;
    int promote = 0, tier = 0;   /* tier: 0 = GPU-baked, 1 = CPU-batch */
    int do_validate = 0, do_validate_gpu = 0, do_bench = 0, smoke = 0, crowd_f16 = 0, cycle_models = 0;
    const char *gltf_path = NULL, *mrw_path = NULL, *bench_out = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--screenshot") && i + 1 < argc) shot = argv[++i];
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--promote")) promote = 1;
        else if (!strcmp(argv[i], "--crowd-tier") && i + 1 < argc) tier = !strcmp(argv[++i], "cpu") ? 1 : 0;
        else if (!strcmp(argv[i], "--crowd-f16")) { crowd_f16 = 1; tier = 1; }  /* f16 is a CPU-tier path */
        else if (!strcmp(argv[i], "--validate")) do_validate = 1;
        else if (!strcmp(argv[i], "--validate-gpu")) do_validate_gpu = 1;
        else if (!strcmp(argv[i], "--bench")) do_bench = 1;
        else if (!strcmp(argv[i], "--bench-out") && i + 1 < argc) bench_out = argv[++i];
        else if (!strcmp(argv[i], "--smoke")) smoke = 1;
        else if (!strcmp(argv[i], "--selftest-assets")) selftest_assets = 1;
        else if (!strcmp(argv[i], "--cycle-models")) cycle_models = 1;  /* headless switch smoke */
        else if (!strcmp(argv[i], "--showcase")) heroes_set_showcase(1); /* CPU pose-ops scene */
        else if (!strcmp(argv[i], "--gltf") && i + 1 < argc) gltf_path = argv[++i];
        else if (!strcmp(argv[i], "--mrw") && i + 1 < argc) mrw_path = argv[++i];
    }
    if (count == 0) { fprintf(stderr, "--count must be >= 1; using 1\n"); count = 1; }
    int capture_frame = max_frames > 0 ? max_frames - 1 : 5;
    if (shot && max_frames == 0) max_frames = capture_frame + 1;

    /* Startup model: the procedural biped unless --gltf overrides it (which also drives the headless
     * --validate/--bench/--screenshot flows). The interactive path additionally discovers every baked
     * model in the assets folder and cycles them with M. The headless paths need only this one. */
    ModelSource startup_src;
    if (gltf_path) model_source_from_gltf(&startup_src, gltf_path, mrw_path);
    else           model_source_procedural(&startup_src);

    int headless = selftest_assets || do_validate || do_validate_gpu || do_bench;
    ProcAssets assets = {0};
    if (headless && load_assets(&startup_src, &assets) != 0) { fprintf(stderr, "asset load failed\n"); return 1; }
    if (selftest_assets) { assets_proc_free(&assets); return 0; }
    if (do_validate) {
        int rc = validate_run(&assets, count >= 4096 ? count : 16384);
        assets_proc_free(&assets);
        return rc;
    }

    if (volkInitialize() != VK_SUCCESS) { fprintf(stderr, "volk: no Vulkan loader\n"); return 1; }
    glfwInitVulkanLoader((PFN_vkGetInstanceProcAddr)vkGetInstanceProcAddr);
    if (!glfwInit()) { fprintf(stderr, "glfw init failed\n"); return 1; }
    if (!glfwVulkanSupported()) { fprintf(stderr, "glfw: Vulkan not supported\n"); return 1; }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    if (do_validate_gpu || do_bench || cycle_models) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  /* hidden-window headless */
    GLFWwindow *win = glfwCreateWindow(1280, 720, "marrow - Vulkan demo", NULL, NULL);
    if (!win) { fprintf(stderr, "window creation failed\n"); return 1; }

    VkCtx ctx;
    if (!vkc_init(&ctx, win)) { fprintf(stderr, "vulkan init failed\n"); return 1; }

    if (do_validate_gpu) {
        int rc = validate_run(&assets, count >= 4096 ? count : 16384);
        rc |= validate_gpu(&ctx, &assets);
        rc |= validate_gpu_f16(&ctx, &assets);
        vkc_wait_idle(&ctx); vkc_destroy(&ctx); assets_proc_free(&assets);
        glfwDestroyWindow(win); glfwTerminate();
        return rc;
    }

    Ground ground;
    if (!ground_init(&ground, &ctx)) { fprintf(stderr, "ground init failed\n"); return 1; }

    if (do_bench) {
        SharedMesh mesh;
        if (!shared_mesh_init(&mesh, &ctx, &assets)) { fprintf(stderr, "shared mesh init failed\n"); return 1; }
        int rc = bench_run(&ctx, &assets, &mesh, &ground, smoke, bench_out);
        vkc_wait_idle(&ctx);
        ground_destroy(&ground, &ctx); shared_mesh_destroy(&mesh, &ctx);
        vkc_destroy(&ctx); assets_proc_free(&assets);
        glfwDestroyWindow(win); glfwTerminate();
        return rc;
    }

    /* interactive: discover the switchable models, then build the GPU resources for the startup one
     * (the procedural biped unless --gltf was given). M cycles through the rest. */
    ModelRegistry reg;
    int model_idx = models_discover(&reg, gltf_path ? &startup_src : NULL);
    fprintf(stderr, "[demo] %d model(s) (M cycles):", reg.count);
    for (int i = 0; i < reg.count; ++i)
        fprintf(stderr, " %s%s", reg.items[i].name, i == model_idx ? "*" : "");
    fprintf(stderr, "\n");

    /* Capacity ceiling for live count adjustment (-/=). Bounded by CROWD_CPU_MAX so the GPU-baked and
     * CPU-batch tiers ALWAYS draw the same instance count - an honest A/B (the CPU tier is hard-capped
     * there anyway; higher GPU-only counts are the headless --bench sweep's job). Buffers are sized to
     * this once per model, so the knob only rebuilds the grid - it never reallocates. */
    uint32_t capacity = CROWD_CPU_MAX;
    if (count > capacity) count = capacity;

    /* Job pool for the CPU-batch tier - the demo schedules marrow's batch across cores (the runtime
     * owns no threads). Auto-sized to the host; shared across model switches (model-independent). */
    Jobs *pool = jobs_create(0);
    if (pool) fprintf(stderr, "[demo] job pool: %u lanes (J toggles jobified CPU-batch)\n",
                      jobs_worker_count(pool));

    ModelGpu mg;
    if (!model_gpu_load(&mg, &ctx, &reg.items[model_idx], capacity, count, crowd_f16, pool)) {
        fprintf(stderr, "model load failed\n"); return 1;
    }
    if (!mg.have_crowd) { tier = 1; fprintf(stderr, "[demo] rig has no BAKED section - CPU-batch crowd only\n"); }

    Hud hud;
    if (!hud_init(&hud, &ctx)) { fprintf(stderr, "hud init failed\n"); return 1; }

    Profiler prof; prof_init(&prof);
    Scene s = { .ctx = &ctx, .prof = &prof, .mesh = &mg.mesh, .ground = &ground,
                .crowd = mg.have_crowd ? &mg.crowd : NULL, .ccpu = &mg.ccpu, .heroes = &mg.heroes,
                .hud = &hud, .joint_count = mg.assets.joint_count, .model = reg.items[model_idx].name };

    Camera cam; camera_init(&cam, v3(0.0f, 8.0f, 40.0f), 0.0f, -0.20f);

    uint64_t rendered = 0;
    int cyc_seen = 1;   /* --cycle-models: models visited so far (startup counts as one) */
    int prev_p = 0, prev_f1 = 0, prev_f2 = 0, prev_t = 0, prev_b = 0, prev_r = 0, prev_j = 0, prev_h = 0, prev_m = 0;
    int prev_dec = 0, prev_inc = 0, prev_k1 = 0, prev_k2 = 0, prev_k3 = 0;
    double prev = prof_now_s(), last_title = prev;
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(win, 1);
        int kp = glfwGetKey(win, GLFW_KEY_P);  if (kp == GLFW_PRESS && !prev_p) promote = !promote; prev_p = kp;
        int kf1 = glfwGetKey(win, GLFW_KEY_F1); if (kf1 == GLFW_PRESS && !prev_f1) hud.visible = !hud.visible; prev_f1 = kf1;
        int kf2 = glfwGetKey(win, GLFW_KEY_F2); if (kf2 == GLFW_PRESS && !prev_f2) hud.detail = !hud.detail; prev_f2 = kf2;
        int kt = glfwGetKey(win, GLFW_KEY_T);
        if (kt == GLFW_PRESS && !prev_t && mg.have_crowd) { tier = !tier; prof_reset_stats(&prof); } prev_t = kt;

        /* M: cycle to the next model. Rebuilds every model-dependent resource (assets, mesh, both
         * crowd tiers, heroes) for the new character while keeping the live count, camera, HUD, job
         * pool, and the CPU-tier config (backend / jobs / f16). The device is idled first so the old
         * buffers are retired before they're freed. On a load failure the previous model is restored. */
        int km = glfwGetKey(win, GLFW_KEY_M);
        int want_next = (km == GLFW_PRESS && !prev_m);
        prev_m = km;
        /* --cycle-models: a headless smoke for the switch path - advance every 20 frames, then quit
         * once every model has been visited once (or immediately if there's only one). */
        if (cycle_models && rendered > 0 && rendered % 20 == 0) {
            if (reg.count > 1 && cyc_seen < reg.count) { want_next = 1; ++cyc_seen; }
            else glfwSetWindowShouldClose(win, 1);
        }
        if (want_next && reg.count > 1) {
            int nidx = (model_idx + 1) % reg.count;
            mrw_backend sv_backend = mg.ccpu.disp.backend;     /* survive the rebuild */
            int sv_jobs = mg.ccpu.threaded, sv_f16 = mg.ccpu.use_f16;
            vkc_wait_idle(&ctx);
            model_gpu_destroy(&mg, &ctx);
            if (model_gpu_load(&mg, &ctx, &reg.items[nidx], capacity, count, sv_f16, pool)) {
                model_idx = nidx;
            } else if (!model_gpu_load(&mg, &ctx, &reg.items[model_idx], capacity, count, sv_f16, pool)) {
                fprintf(stderr, "[demo] fatal: could not reload model '%s'\n", reg.items[model_idx].name);
                break;
            }
            crowd_cpu_set_backend(&mg.ccpu, sv_backend);
            if (pool && mg.ccpu.threaded != sv_jobs) crowd_cpu_toggle_jobs(&mg.ccpu);
            s.mesh = &mg.mesh;
            s.crowd = mg.have_crowd ? &mg.crowd : NULL;
            s.ccpu = &mg.ccpu;
            s.heroes = &mg.heroes;
            s.joint_count = mg.assets.joint_count;
            s.model = reg.items[model_idx].name;
            if (!mg.have_crowd) tier = 1;   /* CPU-only rig: CPU-batch crowd is the only one */
            fprintf(stderr, "[demo] model -> %s (%u joints%s)\n", reg.items[model_idx].name,
                    mg.assets.joint_count, mg.have_crowd ? ", Tier-B" : ", Tier-A only");
            prof_reset_stats(&prof);
        }

        /* Backend: B cycles, 1/2/3 jump straight to scalar/SSE2/AVX2. Stats reset so the steady-state
         * percentiles settle on the new backend without carrying the transition frames. */
        int kb = glfwGetKey(win, GLFW_KEY_B);
        if (kb == GLFW_PRESS && !prev_b) { crowd_cpu_cycle_backend(&mg.ccpu); prof_reset_stats(&prof); } prev_b = kb;
        int k1 = glfwGetKey(win, GLFW_KEY_1) == GLFW_PRESS;
        int k2 = glfwGetKey(win, GLFW_KEY_2) == GLFW_PRESS;
        int k3 = glfwGetKey(win, GLFW_KEY_3) == GLFW_PRESS;
        if ((k1 && !prev_k1) || (k2 && !prev_k2) || (k3 && !prev_k3)) {
            mrw_backend want = (k3 && !prev_k3) ? MRW_BACKEND_AVX2
                             : (k2 && !prev_k2) ? MRW_BACKEND_SSE2 : MRW_BACKEND_SCALAR;
            crowd_cpu_set_backend(&mg.ccpu, want); prof_reset_stats(&prof);
        }
        prev_k1 = k1; prev_k2 = k2; prev_k3 = k3;

        int kr = glfwGetKey(win, GLFW_KEY_R);
        if (kr == GLFW_PRESS && !prev_r) prof_reset_stats(&prof); prev_r = kr;

        /* J: fan the CPU-batch palette-gen across the job pool (or back to single-threaded). */
        int kj = glfwGetKey(win, GLFW_KEY_J);
        if (kj == GLFW_PRESS && !prev_j) { crowd_cpu_toggle_jobs(&mg.ccpu); prof_reset_stats(&prof); } prev_j = kj;

        /* H: flip the CPU-batch palette between f32 and f16 - half the write/upload/fetch. */
        int kh = glfwGetKey(win, GLFW_KEY_H);
        if (kh == GLFW_PRESS && !prev_h) { crowd_cpu_toggle_f16(&mg.ccpu); prof_reset_stats(&prof); } prev_h = kh;

        /* Live entity count: '-' halves, '=' (or numpad +/-) doubles. Buffers are capacity-sized, so
         * this only rebuilds the grid + re-uploads - no reallocation. Idle first since the CPU tier's
         * static instance buffer is rewritten in place. */
        int kdec = glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS;
        int kinc = glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_KP_ADD) == GLFW_PRESS;
        int dec = kdec && !prev_dec, inc = kinc && !prev_inc;
        prev_dec = kdec; prev_inc = kinc;
        if (dec || inc) {
            uint32_t nc = count_step(count, inc, capacity);
            if (nc != count) {
                count = nc;
                vkc_wait_idle(&ctx);
                if (mg.have_crowd) crowd_set_count(&mg.crowd, count);
                crowd_cpu_set_count(&mg.ccpu, count);
                prof_reset_stats(&prof);
            }
        }

        double now = prof_now_s();
        float dt = (float)(now - prev); prev = now;
        camera_update(&cam, win, dt);

        if (shot && rendered == (uint64_t)capture_frame) vkc_request_screenshot(&ctx, shot);

        if (!scene_frame(&s, &cam, dt, tier, promote, 1, 1)) continue;
        rendered++;

        if (now - last_title >= 1.0) {
            char title[160]; hud_title_line(&prof, title, sizeof title);
            glfwSetWindowTitle(win, title);
            last_title = now;
        }
        if (max_frames > 0 && rendered >= (uint64_t)max_frames) glfwSetWindowShouldClose(win, 1);
    }

    vkc_wait_idle(&ctx);
    jobs_destroy(pool);   /* no in-flight gen after the loop; ccpu owns its scratch, not the pool */
    hud_destroy(&hud, &ctx);
    model_gpu_destroy(&mg, &ctx);   /* assets, mesh, both crowd tiers, heroes */
    ground_destroy(&ground, &ctx);
    vkc_destroy(&ctx);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
