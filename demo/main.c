/* marrow Vulkan demo - entry point.
 *
 * Showcases the marrow animation runtime across both tiers and MEASURES it honestly. The interactive
 * scene is a UNIFIED LOD FIELD (field.c): one entity field, each entity classified per frame by
 * distance to the camera into a near CPU "Tier A" band (mrw_batch_blend_clips_to_palette - an exact
 * local-space clip cross-fade) and a far baked-GPU "Tier B" crowd (crowd.c) - the SAME animation,
 * differing only by tier, which is marrow's LOD-promotion contract made visible. The headless paths
 * measure the kernels directly:
 *   - --bench: a GPU-baked + CPU-batch (crowd_cpu.c) sweep over counts/backends with GPU-timestamp
 *     readback and a vertex-skinning microbenchmark;
 *   - --validate / --validate-gpu: CPU parity gates (scalar vs SSE2/AVX2, jobified, f16) + GPU skin.
 * An in-window HUD (F1) shows per-stage CPU/GPU timings and the live LOD split.
 *
 * Controls: WASD/Q/E fly, hold right-mouse to look, Shift sprint, Esc quit.
 *           M cycle model (procedural biped + every baked model in the assets folder),
 *           -/= halve/double the live entity count, [ / ] shrink/grow the Tier-A radius R_A,
 *           ; / ' shrink/grow the render-LOD radius R_mesh, K toggle whole-field bone-line skeleton,
 *           R reset perf stats, F1 toggle HUD, F2 HUD detail.
 * Flags: --count N, --lod-range R_A[,R_mesh], --skeleton, --cam X,Y,Z[,yaw,pitch], --no-hud,
 *        --frames N, --screenshot PATH (deterministic: fixed timestep + fixed camera + no input),
 *        --video DIR (deterministic: dump every frame of a cinematic crowd fly-by as a PNG sequence),
 *        --validate, --validate-gpu, --bench [--bench-out FILE] [--smoke], --selftest-assets,
 *        --cycle-models, --field-smoke, --gltf PATH [--mrw PATH]. */
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
#include "field.h"
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
    int          skeleton;      /* K / --skeleton: draw the GPU-baked crowd as bone lines, not mesh */
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
    int crowd_skel = use_gpu_crowd && s->skeleton;     /* crowd drawn as bone lines, not skinned mesh */
    /* The skeleton render LOD draws lines, not triangles, so it contributes ZERO triangles - but the
     * full per-instance/per-bone ANIMATION still runs, so instances/bones are unchanged. This is the
     * point: render cost drops off the vertex wall while marrow's number holds. */
    uint64_t mesh_inst = (crowd_skel ? 0 : crowd_count) + (draw_heroes ? s->heroes->count : 0);
    prof->instances = inst;
    prof->triangles = tris_per * mesh_inst + 2;     /* + ground quad */
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
    if (use_gpu_crowd && s->skeleton)
        crowd_draw_skeleton(s->crowd, ctx, cmd, &view_proj, extent,
                            s->crowd->inst_addr[ctx->cur_frame], s->crowd->count);
    else if (use_gpu_crowd) crowd_draw(s->crowd, ctx, cmd, &view_proj, extent, s->crowd_mode);
    else                    crowd_cpu_draw(s->ccpu, ctx, cmd, &view_proj, extent);
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

/* Interactive frame for the unified LOD field. GPU zones: CROWD = far Tier-B full-mesh band,
 * HEROES = near Tier-A mesh band, SKEL = all bone-line work (the far skeleton tail past R_mesh, plus
 * the whole near set under the global all-skeleton toggle) - so mesh vs skeleton cost reads separately
 * in the HUD. This is the live scene only; --bench / --validate keep their own paths (crowd + ccpu). */
static int field_frame(VkCtx *ctx, Profiler *prof, Ground *ground, Field *field, Hud *hud,
                       Camera *cam, float dt, const char *model, int draw_hud) {
    VkCommandBuffer cmd; VkExtent2D extent;
    if (!vkc_begin_frame(ctx, &cmd, &extent)) return 0;

    prof_add_cpu(prof, PROF_WAIT_THROTTLE, ctx->wait_ms);
    prof_add_cpu(prof, PROF_ACQUIRE, ctx->acquire_ms);
    float gms[VKC_GPU_ZONE_COUNT];
    uint32_t gmask = vkc_gpu_results_ms(ctx, gms);
    for (uint32_t z = 0; z < VKC_GPU_ZONE_COUNT; ++z)
        if (gmask & (1u << z)) prof_add_gpu(prof, (prof_gpu_zone)z, gms[z]);

    float cam_pos[3] = { cam->pos.x, cam->pos.y, cam->pos.z };
    field_update(field, cam_pos, dt, ctx->cur_frame, prof);

    float aspect = (float)extent.width / (float)extent.height;
    mat4 view_proj = camera_view_proj(cam, aspect);

    /* Publish counters BEFORE recording so the HUD reads THIS frame's split. The render-LOD split:
     * near is full mesh unless the global all-skeleton toggle is on; the far set is a full-mesh band
     * plus a bone-line tail past R_mesh. Triangles count MESH instances only (skeletons are lines,
     * counted separately); the per-instance/per-bone ANIMATION still runs for every entity, so `bones`
     * covers the whole field. The batched-blend kernel (PALETTE_GEN) only touches the near set, so
     * gen_bones is near-only. */
    const SharedMesh *mesh = field->crowd->mesh;
    uint64_t inst = field->count;
    uint64_t tris_per = mesh->index_count / 3;
    int near_skel = field->skeleton_all;
    uint64_t mesh_inst = (uint64_t)field->far_mesh_count + (near_skel ? 0u : field->near_count);
    uint64_t skel_inst = (uint64_t)field->far_skel_count + (near_skel ? field->near_count : 0u);
    uint64_t lines_per = (uint64_t)field->crowd->skel_vert_count / 2u;   /* one LINE_LIST segment per bone */
    prof->instances  = inst;
    prof->triangles  = tris_per * mesh_inst + 2;       /* + ground quad */
    prof->lines      = lines_per * skel_inst;
    prof->bones      = (uint64_t)field->joint_count * inst;
    prof->gen_bones  = (uint64_t)field->joint_count * field->near_count;
    int near_mesh_draw = field->near_count && !near_skel;
    int near_skel_draw = field->near_count &&  near_skel;
    prof->draws      = 1u                                   /* ground */
                     + (field->far_mesh_count ? 1u : 0u)
                     + (near_mesh_draw       ? 1u : 0u)
                     + (field->far_skel_count ? 1u : 0u)
                     + (near_skel_draw       ? 1u : 0u)
                     + (draw_hud             ? 1u : 0u);
    prof->tier       = "field";
    prof->backend    = field->backend_name;
    prof->model      = model;
    prof->field_near     = field->near_count;
    prof->field_far      = field->far_count;
    prof->field_far_mesh = field->far_mesh_count;
    prof->field_far_skel = field->far_skel_count;
    prof->field_clamped  = field->near_clamped;
    prof->field_r_a      = field->lod.r_a;
    prof->field_r_mesh   = field->lod.r_mesh;
    prof->field_skel_all = field->skeleton_all;

    prof_begin(prof, PROF_RECORD);
    vkc_gpu_zone_begin(ctx, cmd, VKC_GPU_ZONE_GROUND);
    ground_draw(ground, ctx, cmd, &view_proj, extent);
    vkc_gpu_zone_end(ctx, cmd, VKC_GPU_ZONE_GROUND);

    vkc_gpu_zone_begin(ctx, cmd, VKC_GPU_ZONE_CROWD);   /* far Tier-B full-mesh band */
    field_draw_far(field, ctx, cmd, &view_proj, extent);
    vkc_gpu_zone_end(ctx, cmd, VKC_GPU_ZONE_CROWD);

    vkc_gpu_zone_begin(ctx, cmd, VKC_GPU_ZONE_HEROES);  /* near Tier-A mesh band */
    field_draw_near(field, ctx, cmd, &view_proj, extent);
    vkc_gpu_zone_end(ctx, cmd, VKC_GPU_ZONE_HEROES);

    vkc_gpu_zone_begin(ctx, cmd, VKC_GPU_ZONE_SKEL);    /* bone-line render LOD: far tail + global near */
    field_draw_far_skeleton(field, ctx, cmd, &view_proj, extent);
    field_draw_near_skeleton(field, ctx, cmd, &view_proj, extent);
    vkc_gpu_zone_end(ctx, cmd, VKC_GPU_ZONE_SKEL);

    if (draw_hud) hud_draw(hud, ctx, cmd, extent, prof);
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

/* ------------------------------------------------------------------ per-model field resources
 * Everything that depends on the loaded character for the interactive field scene: the .mrw assets,
 * the shared skinned mesh, the borrowed Crowd (Tier-B draw machinery), and the unified LOD field.
 * Bundled so the M-key model switch can tear it all down and rebuild it for the next character while
 * the device-, window-, ground-, and HUD-level state lives on. (--bench / --validate build their own
 * crowd + ccpu directly and never touch this.) */
#define FIELD_TOTAL_CAP 65536u   /* Tier B has no 16k cap, so the field scales well past CROWD_CPU_MAX */
#define FIELD_NEAR_CAP   8192u   /* nearest Tier-A (CPU) entities; the rest fall back to Tier B        */

typedef struct {
    ProcAssets assets;
    SharedMesh mesh;
    Crowd      crowd;   /* borrowed by the field for the Tier-B draw machinery + the shared mesh */
    Field      field;
} FieldModel;

static int load_assets(const ModelSource *src, ProcAssets *out) {
    if (src->kind == MODEL_PROCEDURAL) return assets_proc_build(out);
    return assets_gltf_build(src->mrw, src->gltf, out);
}

static void field_model_destroy(FieldModel *m, VkCtx *ctx) {
    /* field + crowd hold marrow views that borrow assets->blob, so free assets last. */
    field_destroy(&m->field, ctx);
    crowd_destroy(&m->crowd, ctx);
    shared_mesh_destroy(&m->mesh, ctx);
    assets_proc_free(&m->assets);
    memset(m, 0, sizeof *m);
}

/* Build the field-scene resources for `src`. Returns 0 with nothing leaked on failure. The character
 * MUST have a BAKED (Tier-B) section - the field's far tier needs it. */
static int field_model_load(FieldModel *m, VkCtx *ctx, const ModelSource *src,
                            uint32_t count, FieldLod lod) {
    memset(m, 0, sizeof *m);
    if (load_assets(src, &m->assets) != 0) {
        fprintf(stderr, "[demo] asset load failed for '%s'\n", src->name); return 0;
    }
    if (!m->assets.tier_b_eligible) {
        fprintf(stderr, "[demo] '%s' has no Tier-B (BAKED) section - the field scene needs it\n", src->name);
        assets_proc_free(&m->assets); return 0;
    }
    if (!shared_mesh_init(&m->mesh, ctx, &m->assets)) {
        fprintf(stderr, "[demo] shared mesh init failed\n");
        assets_proc_free(&m->assets); return 0;
    }
    /* The Crowd is borrowed only for its Tier-B draw path (bindless baked palette + pipeline +
     * crowd_draw_instances) and the shared mesh, so it is built at minimal capacity - the field owns
     * the real far-tier InstanceAnim buffers, sized to FIELD_TOTAL_CAP. */
    if (!crowd_init(&m->crowd, ctx, &m->assets, &m->mesh, 1u, 1u)) {
        fprintf(stderr, "[demo] crowd (Tier-B machinery) init failed\n");
        shared_mesh_destroy(&m->mesh, ctx); assets_proc_free(&m->assets); return 0;
    }
    if (!field_init(&m->field, ctx, &m->assets, &m->crowd, FIELD_TOTAL_CAP, FIELD_NEAR_CAP, count, lod)) {
        fprintf(stderr, "[demo] field init failed\n");
        crowd_destroy(&m->crowd, ctx);
        shared_mesh_destroy(&m->mesh, ctx); assets_proc_free(&m->assets); return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ deterministic capture / smoke
 * The interactive loop advances animation with wall-clock dt and captures by frame number, so a
 * given capture frame does NOT fix phase, blend weight, camera, or the LOD partition. These two
 * headless drivers replace that with a FIXED timestep, a fixed camera, and no input polling, so the
 * field renders reproducibly: the same capture frame is the same simulated time across runs. The
 * per-frame timings the HUD shows are still the real measured ones - only the animation clock is
 * fixed. */

/* Drive `frames` fixed-step field frames from the (fixed) camera, drawing the HUD. If `shot` is set,
 * request the screenshot on the last frame. A swapchain-rebuild skip does not advance the count, so
 * the captured frame is always the same simulated time. */
static void field_capture(VkCtx *ctx, Profiler *prof, Ground *ground, Field *field, Hud *hud,
                          Camera *cam, const char *model, int frames, const char *shot) {
    const float dt = 1.0f / 60.0f;   /* fixed step: reproducible phase / weight / LOD partition */
    if (frames < 1) frames = 1;
    int rendered = 0;
    while (rendered < frames) {
        glfwPollEvents();            /* pump the OS message queue; input is deliberately ignored */
        if (shot && rendered == frames - 1) vkc_request_screenshot(ctx, shot);
        if (field_frame(ctx, prof, ground, field, hud, cam, dt, model, 1)) rendered++;
    }
}

/* Cinematic fly-by pose for the --video capture: a smooth swoop from a high establishing overview
 * down into a low forward skim along the crowd, framed by a moving look-at target so the field stays
 * in shot the whole way. `t` runs 0->1 across the clip and drives the camera entirely (so the capture
 * is reproducible); --cam is ignored in video mode. */
static void flyby_pose(Camera *cam, float t) {
    float s = t * t * (3.0f - 2.0f * t);              /* ease in/out: start and settle gently */
    float u = 1.0f - s;
    float w0 = u * u, w1 = 2.0f * u * s, w2 = s * s;  /* quadratic Bezier weights */
    /* position: high & back -> banked descent to one side -> low forward skim into the crowd */
    vec3 p0 = v3(0.0f, 42.0f, 72.0f), p1 = v3(70.0f, 14.0f, -10.0f), p2 = v3(0.0f, 10.0f, -150.0f);
    vec3 pos = v3_add(v3_add(v3_scale(p0, w0), v3_scale(p1, w1)), v3_scale(p2, w2));
    /* look-at target sweeps down the crowd's length toward the bone-line skeleton horizon */
    vec3 g0 = v3(0.0f, 2.0f, -40.0f), g1 = v3(10.0f, 2.0f, -160.0f), g2 = v3(0.0f, 3.0f, -340.0f);
    vec3 tgt = v3_add(v3_add(v3_scale(g0, w0), v3_scale(g1, w1)), v3_scale(g2, w2));
    vec3 fwd = v3_normalize(v3_sub(tgt, pos));
    float fy = fwd.y < -1.0f ? -1.0f : (fwd.y > 1.0f ? 1.0f : fwd.y);
    cam->pos = pos;
    cam->pitch = asinf(fy);
    cam->yaw   = atan2f(fwd.x, -fwd.z);   /* inverse of camera_forward: fwd=(sin y cos p, sin p, -cos y cos p) */
}

/* Drive `frames` fixed-step field frames along the fly-by, writing each to DIR/frame_NNNNN.png via the
 * swapchain readback. Fixed timestep + camera-from-t + no input makes it reproducible, while the
 * animation clock advances continuously so the crowd moves smoothly across the captured sequence. DIR
 * must already exist (the PNG writer does not create directories). */
static void field_flyby_capture(VkCtx *ctx, Profiler *prof, Ground *ground, Field *field, Hud *hud,
                                Camera *cam, const char *model, int frames, const char *dir) {
    const float dt = 1.0f / 60.0f;   /* fixed step: reproducible phase / weight / LOD partition */
    if (frames < 1) frames = 1;
    int rendered = 0;
    char path[1024];
    while (rendered < frames) {
        glfwPollEvents();            /* pump the OS message queue; input is deliberately ignored */
        float t = frames > 1 ? (float)rendered / (float)(frames - 1) : 0.0f;
        flyby_pose(cam, t);
        snprintf(path, sizeof path, "%s/frame_%05d.png", dir, rendered);
        vkc_request_screenshot(ctx, path);   /* captured + written synchronously inside the next end_frame */
        if (field_frame(ctx, prof, ground, field, hud, cam, dt, model, 0)) rendered++;
    }
}

/* Headless field+skeleton smoke (hidden window, like --cycle-models): drive the unified LOD field
 * for a few fixed-step frames in BOTH render modes so every field draw path is CI-exercised -
 * default (near Tier-A mesh + far Tier-B mesh band + far bone-line tail past R_mesh), then the
 * global all-skeleton toggle (whole field as bone lines). Asserts the per-frame split actually
 * populated each band, so a regression that silently drops a draw fails instead of passing green.
 * The caller sizes the field so all three bands are guaranteed non-empty. Returns 0 on pass. */
static int field_smoke_run(VkCtx *ctx, Profiler *prof, Ground *ground, Field *field, Hud *hud, Camera *cam) {
    const int per_mode = 8;
    int rc = 0;

    /* combined headline: near Tier-A mesh + far Tier-B mesh band + far bone-line tail */
    field->skeleton_all = 0;
    field_capture(ctx, prof, ground, field, hud, cam, "(field-smoke)", per_mode, NULL);
    uint32_t near0 = field->near_count, mesh0 = field->far_mesh_count, skel0 = field->far_skel_count;
    if (near0 == 0) { fprintf(stderr, "[field-smoke] FAIL: no near Tier-A entities\n"); rc = 1; }
    if (mesh0 == 0) { fprintf(stderr, "[field-smoke] FAIL: no far Tier-B mesh band\n"); rc = 1; }
    if (skel0 == 0) { fprintf(stderr, "[field-smoke] FAIL: no far skeleton tail\n"); rc = 1; }

    /* global all-skeleton: the whole far set is the bone-line tail (no far mesh band), near as lines */
    field->skeleton_all = 1;
    field_capture(ctx, prof, ground, field, hud, cam, "(field-smoke)", per_mode, NULL);
    if (field->far_mesh_count != 0)                { fprintf(stderr, "[field-smoke] FAIL: all-skeleton still drew a far mesh band\n"); rc = 1; }
    if (field->far_skel_count != field->far_count) { fprintf(stderr, "[field-smoke] FAIL: all-skeleton far tail != far count\n"); rc = 1; }
    if (field->near_count == 0)                    { fprintf(stderr, "[field-smoke] FAIL: no near entities for the all-skeleton near lines\n"); rc = 1; }

    fprintf(stderr, "[field-smoke] %s | combined: near %u, far mesh %u, far skel %u | all-skeleton: far skel %u\n",
            rc == 0 ? "PASS" : "FAIL", near0, mesh0, skel0, field->far_skel_count);
    return rc;
}

int main(int argc, char **argv) {
    int max_frames = 0; const char *shot = NULL; int selftest_assets = 0;
    uint32_t count = 1600; int count_set = 0;
    int do_validate = 0, do_validate_gpu = 0, do_bench = 0, smoke = 0, cycle_models = 0, field_smoke = 0;
    int skeleton = 0;   /* --skeleton / K: draw the WHOLE field as bone lines (global render mode) */
    int no_hud = 0;     /* --no-hud: start with the HUD hidden (clean captures; deterministic frame) */
    FieldLod lod = { .r_a = 35.0f, .r_mesh = 55.0f };   /* near band Tier-A; mesh out to R_mesh, skeleton tail beyond */
    float cam_arg[5] = { 0, 0, 0, 0, 0 }; int cam_set = 0;   /* --cam X,Y,Z[,yaw,pitch]: reproducible shot framing */
    const char *gltf_path = NULL, *mrw_path = NULL, *bench_out = NULL;
    const char *video_dir = NULL;   /* --video DIR: dump a crowd fly-by as a PNG sequence into DIR */
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--screenshot") && i + 1 < argc) shot = argv[++i];
        else if (!strcmp(argv[i], "--video") && i + 1 < argc) video_dir = argv[++i];
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) { count = (uint32_t)strtoul(argv[++i], NULL, 10); count_set = 1; }
        else if (!strcmp(argv[i], "--lod-range") && i + 1 < argc) {
            /* R_A[,R_mesh] (world units): the animation-tier radius and the optional render-LOD radius */
            float ra = 0.0f, rm = 0.0f;
            int n = sscanf(argv[++i], "%f,%f", &ra, &rm);
            if (n >= 1 && ra > 0.0f) lod.r_a = ra;
            if (n >= 2 && rm > 0.0f) lod.r_mesh = rm;
        }
        else if (!strcmp(argv[i], "--cam") && i + 1 < argc) {
            /* X,Y,Z[,yaw,pitch] (world units / radians): override the camera for a reproducible shot */
            int n = sscanf(argv[++i], "%f,%f,%f,%f,%f", &cam_arg[0], &cam_arg[1], &cam_arg[2], &cam_arg[3], &cam_arg[4]);
            if (n >= 3) cam_set = 1;
        }
        else if (!strcmp(argv[i], "--validate")) do_validate = 1;
        else if (!strcmp(argv[i], "--validate-gpu")) do_validate_gpu = 1;
        else if (!strcmp(argv[i], "--bench")) do_bench = 1;
        else if (!strcmp(argv[i], "--bench-out") && i + 1 < argc) bench_out = argv[++i];
        else if (!strcmp(argv[i], "--smoke")) smoke = 1;
        else if (!strcmp(argv[i], "--selftest-assets")) selftest_assets = 1;
        else if (!strcmp(argv[i], "--cycle-models")) cycle_models = 1;  /* headless switch smoke */
        else if (!strcmp(argv[i], "--field-smoke")) field_smoke = 1;    /* headless field+skeleton smoke */
        else if (!strcmp(argv[i], "--skeleton")) skeleton = 1;
        else if (!strcmp(argv[i], "--no-hud")) no_hud = 1;
        else if (!strcmp(argv[i], "--showcase")) heroes_set_showcase(1); /* CPU pose-ops scene (unwired in P1) */
        else if (!strcmp(argv[i], "--gltf") && i + 1 < argc) gltf_path = argv[++i];
        else if (!strcmp(argv[i], "--mrw") && i + 1 < argc) mrw_path = argv[++i];
    }
    if (count == 0) { fprintf(stderr, "--count must be >= 1; using 1\n"); count = 1; }
    /* A bare --screenshot captures a deterministic frame deep enough that the GPU-timestamp readback
     * is warmed and the HUD frame-time graph (128-sample ring) is full; --frames N overrides. */
    if (shot && max_frames == 0) max_frames = 150;
    /* --video defaults to a ~10-second fly-by at the fixed 1/60 step; --frames overrides the length. */
    if (video_dir && max_frames == 0) max_frames = 600;

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
    if (do_validate_gpu || do_bench || cycle_models || field_smoke) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  /* hidden-window headless */
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

    /* interactive (the unified LOD field): discover the switchable models, then build the field-scene
     * resources for the startup one (the procedural biped unless --gltf was given). M cycles the rest. */
    ModelRegistry reg;
    int model_idx = models_discover(&reg, gltf_path ? &startup_src : NULL);
    fprintf(stderr, "[demo] %d model(s) (M cycles):", reg.count);
    for (int i = 0; i < reg.count; ++i)
        fprintf(stderr, " %s%s", reg.items[i].name, i == model_idx ? "*" : "");
    fprintf(stderr, "\n");

    /* The field's Tier B has no 16k ceiling, so the live-count knob (-/=) runs all the way to
     * FIELD_TOTAL_CAP. Default to a count that already shows BOTH tiers (a near Tier-A band + a large
     * far Tier-B crowd) unless --count overrode it. */
    if (!count_set) count = (shot || video_dir) ? 65536u : 8192u;   /* the headline capture wants a massive far crowd */
    uint32_t capacity = FIELD_TOTAL_CAP;
    if (count > capacity) count = capacity;

    FieldModel mg;
    if (!field_model_load(&mg, &ctx, &reg.items[model_idx], count, lod)) {
        fprintf(stderr, "model load failed\n"); return 1;
    }
    mg.field.skeleton_all = skeleton;   /* --skeleton: start in whole-field bone-line mode */

    Hud hud;
    if (!hud_init(&hud, &ctx)) { fprintf(stderr, "hud init failed\n"); return 1; }
    if (no_hud) hud.visible = 0;   /* clean capture (F1 still toggles it back on interactively) */

    Profiler prof; prof_init(&prof);
    /* Default framing is the headline overview: looking down a vast crowd that recedes into a
     * bone-line skeleton tail at the horizon. --cam overrides it for a different reproducible shot. */
    Camera cam; camera_init(&cam, v3(0.0f, 14.0f, 36.0f), 0.0f, -0.34f);
    if (cam_set) camera_init(&cam, v3(cam_arg[0], cam_arg[1], cam_arg[2]), cam_arg[3], cam_arg[4]);

    int rc = 0;
    if (field_smoke) {
        /* deterministic CI gate: size the field so all three bands populate regardless of the flag
         * defaults (near Tier-A within R_A, a far Tier-B mesh band, and a skeleton tail past R_mesh). */
        uint32_t sc = capacity < 4096u ? capacity : 4096u;
        field_set_count(&mg.field, sc);
        mg.field.lod = (FieldLod){ .r_a = 35.0f, .r_mesh = 90.0f };
        camera_init(&cam, v3(0.0f, 8.0f, 40.0f), 0.0f, -0.20f);
        rc = field_smoke_run(&ctx, &prof, &ground, &mg.field, &hud, &cam);
    } else if (video_dir) {
        /* deterministic fly-by capture: a moving cinematic camera over the massive crowd, every frame
         * dumped as a PNG. The HUD is forced off for a clean reel (F1 can't toggle in headless capture). */
        hud.visible = 0;
        field_flyby_capture(&ctx, &prof, &ground, &mg.field, &hud, &cam,
                            reg.items[model_idx].name, max_frames, video_dir);
    } else if (shot) {
        /* deterministic single-frame capture: fixed timestep + fixed camera + no input polling */
        field_capture(&ctx, &prof, &ground, &mg.field, &hud, &cam, reg.items[model_idx].name, max_frames, shot);
    } else {
    uint64_t rendered = 0;
    int cyc_seen = 1;   /* --cycle-models: models visited so far (startup counts as one) */
    int prev_f1 = 0, prev_f2 = 0, prev_r = 0, prev_m = 0;
    int prev_dec = 0, prev_inc = 0, prev_lb = 0, prev_rb = 0;
    int prev_sc = 0, prev_qt = 0, prev_k = 0;
    double prev = prof_now_s(), last_title = prev;
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(win, 1);
        int kf1 = glfwGetKey(win, GLFW_KEY_F1); if (kf1 == GLFW_PRESS && !prev_f1) hud.visible = !hud.visible; prev_f1 = kf1;
        int kf2 = glfwGetKey(win, GLFW_KEY_F2); if (kf2 == GLFW_PRESS && !prev_f2) hud.detail = !hud.detail; prev_f2 = kf2;

        /* M: cycle to the next model. Rebuilds the field-scene resources (assets, mesh, borrowed crowd,
         * field) for the new character while keeping the live count, camera, HUD, and LOD ranges. The
         * device is idled first so the old buffers retire before they're freed; on a load failure the
         * previous model is restored. */
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
            vkc_wait_idle(&ctx);
            field_model_destroy(&mg, &ctx);
            if (field_model_load(&mg, &ctx, &reg.items[nidx], count, lod)) {
                model_idx = nidx;
            } else if (!field_model_load(&mg, &ctx, &reg.items[model_idx], count, lod)) {
                fprintf(stderr, "[demo] fatal: could not reload model '%s'\n", reg.items[model_idx].name);
                break;
            }
            mg.field.skeleton_all = skeleton;   /* field_init reset it; carry the live toggle across the rebuild */
            fprintf(stderr, "[demo] model -> %s (%u joints)\n", reg.items[model_idx].name, mg.assets.joint_count);
            prof_reset_stats(&prof);
        }

        int kr = glfwGetKey(win, GLFW_KEY_R);
        if (kr == GLFW_PRESS && !prev_r) prof_reset_stats(&prof); prev_r = kr;

        /* [ / ]: shrink / grow R_A (how deep the near Tier-A band reaches); ; / ' : shrink / grow
         * R_mesh (the mesh -> bone-line render-LOD radius). Tracked in `lod` so an M model switch keeps
         * the live ranges, and mirrored onto the field for the next partition. */
        int klb = glfwGetKey(win, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS;
        int krb = glfwGetKey(win, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS;
        int ksc = glfwGetKey(win, GLFW_KEY_SEMICOLON) == GLFW_PRESS;
        int kqt = glfwGetKey(win, GLFW_KEY_APOSTROPHE) == GLFW_PRESS;
        int lod_changed = 0;
        if (klb && !prev_lb) { lod.r_a = lod.r_a > 10.0f ? lod.r_a - 5.0f : 5.0f; lod_changed = 1; }
        if (krb && !prev_rb) { lod.r_a += 5.0f; lod_changed = 1; }
        if (ksc && !prev_sc) { lod.r_mesh = lod.r_mesh > 10.0f ? lod.r_mesh - 5.0f : 5.0f; lod_changed = 1; }
        if (kqt && !prev_qt) { lod.r_mesh += 5.0f; lod_changed = 1; }
        if (lod_changed) { mg.field.lod = lod; prof_reset_stats(&prof); }
        prev_lb = klb; prev_rb = krb; prev_sc = ksc; prev_qt = kqt;

        /* K: toggle the global all-skeleton render mode (whole field as bone lines). */
        int kk = glfwGetKey(win, GLFW_KEY_K) == GLFW_PRESS;
        if (kk && !prev_k) { skeleton = !skeleton; mg.field.skeleton_all = skeleton; prof_reset_stats(&prof); }
        prev_k = kk;

        /* Live entity count: '-' halves, '=' (or numpad +/-) doubles, up to FIELD_TOTAL_CAP. Buffers
         * are capacity-sized, so this only re-lays-out the grid (CPU) - no reallocation, no device
         * idle (the per-frame near/far buffers are re-staged by the next field_update). */
        int kdec = glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS;
        int kinc = glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_KP_ADD) == GLFW_PRESS;
        int dec = kdec && !prev_dec, inc = kinc && !prev_inc;
        prev_dec = kdec; prev_inc = kinc;
        if (dec || inc) {
            uint32_t nc = count_step(count, inc, capacity);
            if (nc != count) { count = nc; field_set_count(&mg.field, count); prof_reset_stats(&prof); }
        }

        double now = prof_now_s();
        float dt = (float)(now - prev); prev = now;
        camera_update(&cam, win, dt);

        if (!field_frame(&ctx, &prof, &ground, &mg.field, &hud, &cam, dt, reg.items[model_idx].name, 1)) continue;
        rendered++;

        if (now - last_title >= 1.0) {
            char title[160]; hud_title_line(&prof, title, sizeof title);
            glfwSetWindowTitle(win, title);
            last_title = now;
        }
        if (max_frames > 0 && rendered >= (uint64_t)max_frames) glfwSetWindowShouldClose(win, 1);
    }
    }   /* interactive loop (else: the deterministic --screenshot / --field-smoke paths above) */

    vkc_wait_idle(&ctx);
    hud_destroy(&hud, &ctx);
    field_model_destroy(&mg, &ctx);   /* field, borrowed crowd, mesh, assets */
    ground_destroy(&ground, &ctx);
    vkc_destroy(&ctx);
    glfwDestroyWindow(win);
    glfwTerminate();
    return rc;
}
