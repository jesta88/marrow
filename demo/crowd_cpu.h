/* Live CPU-batch crowd tier - the flagship CPU batch path, exercised in the render loop.
 *
 * The baked GPU crowd (crowd.c) animates entirely on the GPU (sampling the baked palette texture), so
 * marrow's headline CPU claim - mrw_batch_clip_to_palette scaling tens of thousands of entities
 * across-instance SoA - is never driven by the demo. This tier fixes that: every frame it runs the
 * public zero-alloc batch kernel on a runtime-selectable backend (scalar / SSE2 / AVX2), writing
 * the canonical 3x4 palette, and skins via the SAME shader as the CPU heroes (skin_tierA.vert).
 *
 * Measurement honesty:
 *   - instances are GROUPED BY CLIP and dispatched one homogeneous batch per clip group, a faithful
 *     A/B against the GPU crowd's per-instance clip mix (sorted once at init into contiguous ranges);
 *   - the batch writes a 64-byte-aligned CACHEABLE CPU scratch (timed PALETTE_GEN = true library
 *     throughput) and a SEPARATE memcpy stages it into the mapped GPU SSBO (timed PALETTE_UPLOAD),
 *     so the SIMD comparison is the kernels, not write-combined memory.
 *
 * Capped at CROWD_CPU_MAX; the GPU-baked tier sweeps higher. Uses only the public marrow batch API
 * with caller-owned scratch - the zero-dep runtime is untouched. */
#ifndef DEMO_CROWD_CPU_H
#define DEMO_CROWD_CPU_H

#include "vk_context.h"
#include "assets_proc.h"
#include "linalg.h"
#include "marrow.h"
#include "crowd.h"       /* SharedMesh */
#include "profiler.h"
#include "jobs.h"        /* optional thread pool for jobified palette-gen */

#define CROWD_CPU_MAX 16384u

typedef struct { uint32_t clip, start, count; } CpuClipGroup;  /* contiguous, sorted by clip */

typedef struct {
    VkShaderEXT vs, fs; VkPipelineLayout layout;
    VkShaderEXT vs_f16, fs_f16;      /* f16 palette variant (linked pair; shares the layout) */
    const SharedMesh *mesh;          /* borrowed */

    /* marrow views borrow assets->blob (kept alive for the lifetime) */
    mrw_blob          blob;
    mrw_skeleton_view skel;
    mrw_clip_view     clip[DEMO_PROC_MAX_CLIPS];
    float             clip_dur[DEMO_PROC_MAX_CLIPS];  /* per-clip duration (s); kept so a live count
                                                         change can re-phase without `assets` */
    uint32_t          clip_count, joint_count, count, capacity;

    CpuClipGroup      groups[DEMO_PROC_MAX_CLIPS];
    uint32_t          group_count;
    float            *time;          /* capacity-sized; first `count` live, sorted order, advanced each frame */

    /* static per-instance model + tint (grid never moves; uploaded once) */
    VkBuffer hero_buf; VkDeviceMemory hero_mem; void *hero_map; VkDeviceAddress hero_addr;

    /* per-frame palette SSBO (BDA) the shader fetches */
    VkBuffer pal_buf[VKC_FRAMES_IN_FLIGHT]; VkDeviceMemory pal_mem[VKC_FRAMES_IN_FLIGHT];
    void *pal_map[VKC_FRAMES_IN_FLIGHT];    VkDeviceAddress pal_addr[VKC_FRAMES_IN_FLIGHT];

    /* per-frame f16 palette SSBO (Lever A: 24 B/joint, half the f32 write/upload/fetch). Always
     * allocated so the f32<->f16 A/B (use_f16) is an instant runtime toggle. */
    VkBuffer pal16_buf[VKC_FRAMES_IN_FLIGHT]; VkDeviceMemory pal16_mem[VKC_FRAMES_IN_FLIGHT];
    void *pal16_map[VKC_FRAMES_IN_FLIGHT];    VkDeviceAddress pal16_addr[VKC_FRAMES_IN_FLIGHT];
    size_t pal16_bytes;              /* capacity-sized f16 output bytes (MRW_PALETTE_F16 query) */
    int    use_f16;                  /* drive mrw_batch_clip_to_palette_f16 + the f16 shader */

    /* 64-aligned cacheable CPU scratch (mrw_authoring_alloc): batch output + internal SoA tile(s).
     * The batch scratch holds worker_count contiguous 64-aligned slices so each job lane runs the
     * kernel against its OWN scratch - the per-worker isolation that makes the fan-out thread-safe. */
    float  *pal_scratch;   size_t pal_bytes;
    void   *batch_scratch; size_t batch_bytes;   /* == worker_count * batch_unit */
    size_t  batch_unit;                           /* per-worker scratch bytes (joint_count-derived) */

    mrw_dispatch disp;               /* runtime-selectable backend */
    char         backend_label[24];  /* "AVX2" serial, "AVX2 x8" jobified - shown in the HUD */
    const char  *backend_name;       /* -> backend_label */

    /* Jobified palette-gen. marrow owns no threads; the demo schedules the batch across cores -
     * read-only views/dispatch shared, each lane writing a disjoint output slice with its own scratch. */
    Jobs    *pool;          /* borrowed (owned by main); NULL ⇒ serial only */
    uint32_t worker_count;  /* batch-scratch slices == pool lanes, or 1 when serial */
    int      threaded;      /* fan PALETTE_GEN across the pool */
} CrowdCpu;

/* Allocate scratch + GPU buffers for `capacity` instances (clamped to CROWD_CPU_MAX); start with
 * `count` (<= capacity) live. */
int  crowd_cpu_init(CrowdCpu *cc, VkCtx *ctx, const ProcAssets *assets,
                    const SharedMesh *mesh, uint32_t capacity, uint32_t count);
/* Change the live instance count (clamped to [1, capacity]) and rebuild the grid + clip grouping.
 * Rewrites the static per-instance buffer in place, so the caller must have idled the device first. */
void crowd_cpu_set_count(CrowdCpu *cc, uint32_t count);
void crowd_cpu_set_backend(CrowdCpu *cc, mrw_backend backend);  /* falls back to scalar if unsupported */
void crowd_cpu_cycle_backend(CrowdCpu *cc);                     /* scalar -> SSE2 -> AVX2 -> ... (host-supported) */
/* Attach a (borrowed) thread pool and size the per-worker batch scratch to its lane count; enables
 * jobified palette-gen by default when the pool has >1 lane. Call once after init (CPU-only - no
 * device interaction). Safe to pass NULL (stays serial). */
void crowd_cpu_set_jobs(CrowdCpu *cc, Jobs *pool);
void crowd_cpu_toggle_jobs(CrowdCpu *cc);                       /* no-op without a multi-lane pool */
/* Select the palette output format: 0 = f32 (48 B/joint), 1 = f16 (24 B/joint, Lever A). Both
 * resource sets are pre-allocated, so this just flips which buffer+shader the next frame uses -
 * no device idle, no reallocation. */
void crowd_cpu_set_f16(CrowdCpu *cc, int on);
void crowd_cpu_toggle_f16(CrowdCpu *cc);
/* Advance phases, run the batch into CPU scratch (PALETTE_GEN), upload to the SSBO (PALETTE_UPLOAD). */
void crowd_cpu_update(CrowdCpu *cc, float dt, uint32_t frame, Profiler *prof);
void crowd_cpu_draw(CrowdCpu *cc, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent);
void crowd_cpu_destroy(CrowdCpu *cc, VkCtx *ctx);

#endif /* DEMO_CROWD_CPU_H */
