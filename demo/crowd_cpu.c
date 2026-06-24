#include "crowd_cpu.h"

#include "mrw_authoring.h"   /* mrw_authoring_alloc/free - 64-aligned cacheable CPU scratch */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

/* Mirrors skin_tierA.vert's HeroInstance (scalar layout): mat4 model + vec4 tint. */
typedef struct { float model[16]; float tint[4]; } CpuInstance;
_Static_assert(sizeof(CpuInstance) == 80, "CpuInstance must be 80 bytes");

/* Mirrors skin_tierA.vert's push constant (== HeroPush in heroes.c). */
typedef struct {
    float           viewProj[16];
    VkDeviceAddress heroes;
    VkDeviceAddress palettes;
    uint32_t        jointCount;
    uint32_t        _pad;
} CpuPush;
_Static_assert(sizeof(CpuPush) == 88, "CpuPush must be 88 bytes");

#include "skin_tierA_vert.spv.h"
#include "skin_tierA_f16_vert.spv.h"   /* f16 palette variant (unpackHalf2x16 fetch) */
#include "crowd_frag.spv.h"   /* reuse the crowd fragment (same vNormal/vColor inputs) */

static const char *backend_name(mrw_backend b) {
    return b == MRW_BACKEND_AVX2 ? "AVX2" : (b == MRW_BACKEND_SSE2 ? "SSE2" : "scalar");
}

/* Rebuild the HUD/bench label from backend + threading state. ASCII only (the HUD font is). When
 * jobified it reads e.g. "AVX2 x8"; serial stays the plain "AVX2" the bench self-check matches. */
static void refresh_label(CrowdCpu *cc) {
    const char *bn = backend_name(cc->disp.backend);
    if (cc->threaded && cc->worker_count > 1)
        snprintf(cc->backend_label, sizeof cc->backend_label, "%s x%u", bn, cc->worker_count);
    else
        snprintf(cc->backend_label, sizeof cc->backend_label, "%s", bn);
    cc->backend_name = cc->backend_label;
}

/* crowd.vert's per-instance tint, computed CPU-side so the CPU tier colors match the GPU crowd. */
static void crowd_tint(uint32_t id, float out[4]) {
    out[0] = 0.45f + 0.45f * cosf((float)id * 0.91f);
    out[1] = 0.45f + 0.45f * cosf((float)id * 1.73f + 2.0f);
    out[2] = 0.45f + 0.45f * cosf((float)id * 2.21f + 4.0f);
    out[3] = 1.0f;
}

/* (Re)build the layout for the current cc->count: the grid identical to the baked GPU crowd
 * (crowd.c init_instances) so the A/B is visually equal, a counting sort by clip into contiguous
 * ranges (group_count <= clip_count), per-instance model+tint scattered into the SORTED slot of the
 * mapped static buffer, and the initial time phase. Writes the static GPU buffer in place, so any
 * caller other than init must have idled the device first. */
static void cpu_layout(CrowdCpu *cc) {
    uint32_t count = cc->count;
    uint32_t cols = (uint32_t)ceilf(sqrtf((float)count));
    float spacing = 1.6f;
    float ox = -0.5f * (float)(cols - 1) * spacing;
    float oz = -8.0f - (float)(cols - 1) * spacing;

    uint32_t counts[DEMO_PROC_MAX_CLIPS] = { 0 };
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t cz = i / cols;
        counts[(i * 7u + cz) % cc->clip_count]++;
    }
    uint32_t cursor[DEMO_PROC_MAX_CLIPS], acc = 0;
    cc->group_count = 0;
    for (uint32_t k = 0; k < cc->clip_count; ++k) {
        cursor[k] = acc;
        if (counts[k]) {
            cc->groups[cc->group_count].clip  = k;
            cc->groups[cc->group_count].start = acc;
            cc->groups[cc->group_count].count = counts[k];
            cc->group_count++;
        }
        acc += counts[k];
    }

    CpuInstance *inst = (CpuInstance *)cc->hero_map;   /* persistently mapped host buffer */
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t cx = i % cols, cz = i / cols;
        uint32_t clip = (i * 7u + cz) % cc->clip_count;
        uint32_t dst = cursor[clip]++;
        mat4 m = mat4_translate(v3(ox + cx * spacing, 0.0f, oz + cz * spacing));
        memcpy(inst[dst].model, m.m, sizeof m.m);
        crowd_tint(i, inst[dst].tint);
        float d = cc->clip_dur[clip] > 0.0f ? cc->clip_dur[clip] : 1.0f;
        cc->time[dst] = fmodf((float)i * 0.137f, d);
    }
}

int crowd_cpu_init(CrowdCpu *cc, VkCtx *ctx, const ProcAssets *assets,
                   const SharedMesh *mesh, uint32_t capacity, uint32_t count) {
    memset(cc, 0, sizeof *cc);
    if (capacity > CROWD_CPU_MAX) capacity = CROWD_CPU_MAX;
    if (capacity < 1) capacity = 1;
    if (count > capacity) count = capacity;
    if (count < 1) count = 1;
    cc->capacity = capacity;
    cc->count = count;
    cc->mesh = mesh;
    cc->joint_count = assets->joint_count;

    if (mrw_blob_open(assets->blob, assets->blob_size, &cc->blob) != MRW_OK) {
        fprintf(stderr, "[crowd-cpu] blob open\n"); goto fail;
    }
    mrw_blob_skeleton(&cc->blob, &cc->skel);

    /* Resolve the clip set (CLIP sections - CPU animation, no BAKED needed). */
    cc->clip_count = assets->clip_count < DEMO_PROC_MAX_CLIPS ? assets->clip_count
                                                              : (uint32_t)DEMO_PROC_MAX_CLIPS;
    for (uint32_t k = 0; k < cc->clip_count; ++k) {
        if (mrw_clip_view_at(&cc->blob, 1 + assets->clips[k].clip_index, &cc->clip[k]) != MRW_OK) {
            fprintf(stderr, "[crowd-cpu] clip %u view failed\n", k); goto fail;
        }
        cc->clip_dur[k] = assets->clips[k].duration_s;
    }
    if (cc->clip_count == 0) { fprintf(stderr, "[crowd-cpu] no clips\n"); goto fail; }

    uint32_t jc = cc->joint_count;
    cc->time = (float *)malloc((size_t)capacity * sizeof(float));
    if (!cc->time) goto fail;

    /* batch sizing for the full capacity - palette out + internal SoA tile (both 64-aligned
     * cacheable CPU RAM). The per-frame batch only fills the live `count` prefix of pal_scratch. */
    mrw_mem_req sreq, preq, preq16;
    if (mrw_batch_clip_to_palette_requirements(jc, capacity, MRW_PALETTE_F32, &sreq, &preq) != MRW_OK ||
        mrw_batch_clip_to_palette_requirements(jc, capacity, MRW_PALETTE_F16, NULL, &preq16) != MRW_OK) {
        fprintf(stderr, "[crowd-cpu] batch requirements failed\n"); goto fail;
    }
    /* One scratch slice per lane; start serial (1 lane). crowd_cpu_set_jobs grows this to the pool's
     * lane count. batch_unit is a multiple of 64 (the requirements query rounds up), so every
     * worker*batch_unit slice stays 64-aligned off the 64-aligned base. */
    cc->pal_bytes = preq.size; cc->pal16_bytes = preq16.size; cc->batch_unit = sreq.size;
    cc->worker_count = 1; cc->threaded = 0; cc->pool = NULL;
    cc->batch_bytes = cc->batch_unit;
    /* One cacheable CPU output buffer, sized for the wider f32 entry (48 B/joint); the f16 path reuses
     * it via a uint16_t* view (24 B/joint, fits with room to spare). */
    cc->pal_scratch   = (float *)mrw_authoring_alloc(cc->pal_bytes);
    cc->batch_scratch = mrw_authoring_alloc(cc->batch_bytes);
    if (!cc->pal_scratch || !cc->batch_scratch) { fprintf(stderr, "[crowd-cpu] scratch alloc failed\n"); goto fail; }

    /* GPU buffers sized to capacity: static per-instance model+tint (rewritten by cpu_layout on a
     * live count change), per-frame palette SSBO (BDA). */
    VkBufferUsageFlags su = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (!vkc_create_host_buffer(ctx, (VkDeviceSize)capacity * sizeof(CpuInstance), su,
                                &cc->hero_buf, &cc->hero_mem, &cc->hero_map)) goto fail;
    cc->hero_addr = vkc_buffer_address(ctx, cc->hero_buf);

    for (uint32_t f = 0; f < VKC_FRAMES_IN_FLIGHT; ++f) {
        if (!vkc_create_host_buffer(ctx, (VkDeviceSize)cc->pal_bytes, su,
                                    &cc->pal_buf[f], &cc->pal_mem[f], &cc->pal_map[f])) goto fail;
        cc->pal_addr[f] = vkc_buffer_address(ctx, cc->pal_buf[f]);
        if (!vkc_create_host_buffer(ctx, (VkDeviceSize)cc->pal16_bytes, su,
                                    &cc->pal16_buf[f], &cc->pal16_mem[f], &cc->pal16_map[f])) goto fail;
        cc->pal16_addr[f] = vkc_buffer_address(ctx, cc->pal16_buf[f]);
    }

    cpu_layout(cc);   /* model+tint into the mapped static buffer + clip grouping + time phases */

    VkPushConstantRange pr = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(CpuPush) };
    VkPipelineLayoutCreateInfo li = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(ctx->device, &li, NULL, &cc->layout) != VK_SUCCESS) goto fail;
    if (!vkc_create_graphics_shaders(ctx, skin_tierA_vert_spv, sizeof skin_tierA_vert_spv,
            crowd_frag_spv, sizeof crowd_frag_spv, NULL, 0, &pr, &cc->vs, &cc->fs)) goto fail;
    /* f16 palette variant: same fragment source + push-constant layout, only the vertex palette fetch
     * differs. Linked as its own vs+fs pair - a linked fragment shader may only be bound with the
     * vertex shader it was linked with (VK_EXT_shader_object), so the f16 vs needs its own fs. */
    if (!vkc_create_graphics_shaders(ctx, skin_tierA_f16_vert_spv, sizeof skin_tierA_f16_vert_spv,
            crowd_frag_spv, sizeof crowd_frag_spv, NULL, 0, &pr, &cc->vs_f16, &cc->fs_f16)) goto fail;

    mrw_dispatch_detect(&cc->disp);
    refresh_label(cc);
    fprintf(stderr, "[crowd-cpu] %u instances (capacity %u), %u clip group(s), backend %s\n",
            count, capacity, cc->group_count, cc->backend_name);
    return 1;
fail:   /* let destroy unwind every partially-created resource */
    crowd_cpu_destroy(cc, ctx);
    return 0;
}

void crowd_cpu_set_count(CrowdCpu *cc, uint32_t count) {
    if (count < 1) count = 1;
    if (count > cc->capacity) count = cc->capacity;
    cc->count = count;
    cpu_layout(cc);   /* rewrites the static per-instance buffer in place - caller idled the device */
}

void crowd_cpu_set_backend(CrowdCpu *cc, mrw_backend backend) {
    mrw_dispatch d;
    mrw_result r = backend == MRW_BACKEND_AVX2 ? mrw_dispatch_avx2(&d)
                 : backend == MRW_BACKEND_SSE2 ? mrw_dispatch_sse2(&d)
                                               : mrw_dispatch_scalar(&d);
    if (r != MRW_OK) mrw_dispatch_scalar(&d);   /* unsupported ISA -> scalar */
    cc->disp = d;
    refresh_label(cc);
}

void crowd_cpu_cycle_backend(CrowdCpu *cc) {
    static const mrw_backend order[3] = { MRW_BACKEND_SCALAR, MRW_BACKEND_SSE2, MRW_BACKEND_AVX2 };
    int cur = 0;
    for (int i = 0; i < 3; ++i) if (order[i] == cc->disp.backend) cur = i;
    for (int step = 1; step <= 3; ++step) {
        mrw_backend next = order[(cur + step) % 3];
        mrw_dispatch d;
        mrw_result r = next == MRW_BACKEND_AVX2 ? mrw_dispatch_avx2(&d)
                     : next == MRW_BACKEND_SSE2 ? mrw_dispatch_sse2(&d)
                                                : mrw_dispatch_scalar(&d);
        if (r == MRW_OK) { cc->disp = d; refresh_label(cc); return; }
    }
}

void crowd_cpu_set_jobs(CrowdCpu *cc, Jobs *pool) {
    cc->pool = pool;
    uint32_t lanes = pool ? jobs_worker_count(pool) : 1u;
    if (lanes < 1) lanes = 1;

    if (lanes != cc->worker_count) {
        /* Grow the batch scratch to one 64-aligned slice per lane. On failure stay serial rather
         * than run lanes against overlapping scratch (which would corrupt the SoA tile). */
        size_t want = cc->batch_unit * lanes;
        void *ns = mrw_authoring_alloc(want);
        if (ns) {
            mrw_authoring_free(cc->batch_scratch);
            cc->batch_scratch = ns; cc->batch_bytes = want; cc->worker_count = lanes;
        } else {
            fprintf(stderr, "[crowd-cpu] per-worker scratch alloc failed; staying serial\n");
            cc->pool = NULL;
        }
    }
    cc->threaded = (cc->pool && cc->worker_count > 1) ? 1 : 0;
    refresh_label(cc);
}

void crowd_cpu_toggle_jobs(CrowdCpu *cc) {
    if (!cc->pool || cc->worker_count <= 1) return;   /* single core: nothing to toggle */
    cc->threaded = !cc->threaded;
    refresh_label(cc);
}

/* Both the f32 and f16 GPU resources are pre-allocated and the per-frame buffer for the current
 * frame index is already fenced by vkc_begin_frame, so flipping the format needs no device idle -
 * the next frame simply generates + uploads + binds the other set. */
void crowd_cpu_set_f16(CrowdCpu *cc, int on) { cc->use_f16 = on ? 1 : 0; }
void crowd_cpu_toggle_f16(CrowdCpu *cc) { cc->use_f16 = !cc->use_f16; }

/* Palette-gen for one job lane's instance range [begin,end). The clip groups are sorted, contiguous
 * and disjoint, so intersecting the lane's range with each group yields homogeneous sub-batches whose
 * OUTPUT rows are disjoint across lanes (instance i, joint j lives at (i*jc+j)*12). Each lane runs the
 * SAME public mrw_batch_clip_to_palette - sharing the read-only skeleton/clip views and dispatch, and
 * using ITS OWN scratch slice - which is exactly why the runtime is safe to fan across cores. */
typedef struct { CrowdCpu *cc; uint32_t jc; mrw_result err[JOBS_MAX_WORKERS]; } GenCtx;

static void gen_range(void *vctx, uint32_t worker, uint32_t begin, uint32_t end) {
    GenCtx *g = (GenCtx *)vctx; CrowdCpu *cc = g->cc; uint32_t jc = g->jc;
    void *scratch = (char *)cc->batch_scratch + (size_t)worker * cc->batch_unit;
    for (uint32_t gi = 0; gi < cc->group_count; ++gi) {
        const CpuClipGroup *grp = &cc->groups[gi];
        uint32_t gs = grp->start, ge = grp->start + grp->count;
        uint32_t s = gs > begin ? gs : begin;
        uint32_t e = ge < end   ? ge : end;
        if (e <= s) continue;
        size_t off = (size_t)s * jc * 12u;   /* element offset into the AoS palette (12 comps/joint) */
        mrw_result r;
        if (cc->use_f16)
            /* f16 entry = 24 B/joint; write into the shared pal_scratch as a uint16_t view, and bound
             * the capacity against the (smaller) f16 buffer size so the kernel sees the real f16 span. */
            r = mrw_batch_clip_to_palette_f16(&cc->disp, &cc->skel, &cc->clip[grp->clip],
                    &cc->time[s], e - s,
                    (uint16_t *)cc->pal_scratch + off, cc->pal16_bytes - off * sizeof(uint16_t),
                    scratch, cc->batch_unit);
        else
            r = mrw_batch_clip_to_palette(&cc->disp, &cc->skel, &cc->clip[grp->clip],
                    &cc->time[s], e - s,
                    cc->pal_scratch + off, cc->pal_bytes - off * sizeof(float),
                    scratch, cc->batch_unit);
        if (r != MRW_OK && g->err[worker] == MRW_OK) g->err[worker] = r;
    }
}

void crowd_cpu_update(CrowdCpu *cc, float dt, uint32_t frame, Profiler *prof) {
    uint32_t jc = cc->joint_count;

    prof_begin(prof, PROF_CROWD_UPDATE);
    for (uint32_t i = 0; i < cc->count; ++i) cc->time[i] += dt;
    prof_end(prof, PROF_CROWD_UPDATE);

    /* PALETTE_GEN: homogeneous batch(es) into cacheable CPU RAM (library throughput). Serial runs one
     * lane over the whole crowd; jobified fans contiguous instance ranges across the pool (the marrow
     * runtime owns no threads - the demo is the scheduler). Timed as wall-clock either way, so the HUD
     * shows the real speedup. */
    GenCtx g; g.cc = cc; g.jc = jc;
    for (uint32_t w = 0; w < cc->worker_count; ++w) g.err[w] = MRW_OK;
    prof_begin(prof, PROF_PALETTE_GEN);
    if (cc->threaded && cc->pool && cc->worker_count > 1)
        jobs_parallel_for(cc->pool, cc->count, gen_range, &g);
    else
        gen_range(&g, 0, 0, cc->count);
    prof_end(prof, PROF_PALETTE_GEN);
    for (uint32_t w = 0; w < cc->worker_count; ++w)
        if (g.err[w] != MRW_OK) { fprintf(stderr, "[crowd-cpu] batch lane %u failed: %u\n", w, g.err[w]); break; }

    /* PALETTE_UPLOAD: stage the live range into the write-combined mapped SSBO, separately. Only
     * the first `count` instances are drawn, so don't pay to copy the unused capacity tail. f16 moves
     * half the bytes (12 comps/joint × 2 vs 4) - the Lever A upload-bandwidth win, visible on the HUD. */
    prof_begin(prof, PROF_PALETTE_UPLOAD);
    size_t live_comps = (size_t)cc->count * jc * 12u;
    if (cc->use_f16)
        memcpy(cc->pal16_map[frame], cc->pal_scratch, live_comps * sizeof(uint16_t));
    else
        memcpy(cc->pal_map[frame], cc->pal_scratch, live_comps * sizeof(float));
    prof_end(prof, PROF_PALETTE_UPLOAD);
}

void crowd_cpu_draw(CrowdCpu *cc, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent) {
    VkVertexInputBindingDescription2EXT b = { VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT };
    b.binding = 0; b.stride = sizeof(DemoVertex); b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; b.divisor = 1;
    VkVertexInputAttributeDescription2EXT a[5];
    for (int i = 0; i < 5; ++i) { a[i] = (VkVertexInputAttributeDescription2EXT){ VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT }; a[i].binding = 0; a[i].location = (uint32_t)i; }
    a[0].format = VK_FORMAT_R32G32B32_SFLOAT;    a[0].offset = offsetof(DemoVertex, pos);
    a[1].format = VK_FORMAT_R32G32B32_SFLOAT;    a[1].offset = offsetof(DemoVertex, nrm);
    a[2].format = VK_FORMAT_R32G32B32A32_SFLOAT; a[2].offset = offsetof(DemoVertex, tan);
    a[3].format = VK_FORMAT_R32G32B32A32_UINT;   a[3].offset = offsetof(DemoVertex, bones);
    a[4].format = VK_FORMAT_R32G32B32A32_SFLOAT; a[4].offset = offsetof(DemoVertex, weights);

    if (cc->use_f16) vkc_bind_shaders(ctx, cmd, cc->vs_f16, cc->fs_f16);
    else             vkc_bind_shaders(ctx, cmd, cc->vs, cc->fs);
    vkc_set_default_state(ctx, cmd, extent, &b, 1, a, 5);
    vkCmdSetCullMode(cmd, cc->mesh->cull_back ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE);

    CpuPush pc;
    memcpy(pc.viewProj, view_proj->m, sizeof pc.viewProj);
    pc.heroes = cc->hero_addr;
    pc.palettes = cc->use_f16 ? cc->pal16_addr[ctx->cur_frame] : cc->pal_addr[ctx->cur_frame];
    pc.jointCount = cc->joint_count;
    pc._pad = 0;
    vkCmdPushConstants(cmd, cc->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof pc, &pc);

    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cc->mesh->vbuf, &zero);
    vkCmdBindIndexBuffer(cmd, cc->mesh->ibuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, cc->mesh->index_count, cc->count, 0, 0, 0);
}

void crowd_cpu_destroy(CrowdCpu *cc, VkCtx *ctx) {
    vkDeviceWaitIdle(ctx->device);
    vkc_destroy_shader(ctx, cc->vs); vkc_destroy_shader(ctx, cc->fs);
    vkc_destroy_shader(ctx, cc->vs_f16); vkc_destroy_shader(ctx, cc->fs_f16);
    if (cc->layout) vkDestroyPipelineLayout(ctx->device, cc->layout, NULL);
    vkc_destroy_buffer(ctx, cc->hero_buf, cc->hero_mem);
    for (uint32_t f = 0; f < VKC_FRAMES_IN_FLIGHT; ++f) {
        vkc_destroy_buffer(ctx, cc->pal_buf[f], cc->pal_mem[f]);
        vkc_destroy_buffer(ctx, cc->pal16_buf[f], cc->pal16_mem[f]);
    }
    mrw_authoring_free(cc->pal_scratch);
    mrw_authoring_free(cc->batch_scratch);
    free(cc->time);
    memset(cc, 0, sizeof *cc);
}
