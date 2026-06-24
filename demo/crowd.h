/* Baked GPU crowd: uploads the .mrw BAKED palette stream to an RGBA16F texture (bindless),
 * holds per-instance animation state in a buffer-device-address storage buffer, and issues one
 * instanced draw of the skinned mesh. The headline crowd path. */
#ifndef DEMO_CROWD_H
#define DEMO_CROWD_H

#include "vk_context.h"
#include "assets_proc.h"
#include "linalg.h"

/* Per-instance animation state. Mirrored by the GLSL InstanceAnim under scalar layout - the
 * _Static_assert in crowd.c guards the match. */
typedef struct {
    float    model[16];
    uint32_t clipA[4];   /* first_frame, frame_count, looping, paletteIndex */
    uint32_t clipB[4];   /* first_frame, frame_count, looping, entity id (drives the stable tint) */
    float    times[4];   /* tA, tB, durA, durB                              */
    float    blend[4];   /* w, _, _, _                                      */
} InstanceAnim;

typedef struct {
    uint32_t first_frame, frame_count, looping;
    float    duration;
} CrowdClip;

/* Crowd draw mode. FULL is the real render; the two DISCARD modes are the bench-only
 * vertex-skinning microbenchmark (rasterizer discard on, no fragment stage) - SKIN runs the real
 * baked-palette skinning VS, STATIC runs the matched no-palette baseline, and gpu_skin - gpu_static
 * isolates the marrow palette sample+decode+blend. */
typedef enum {
    CROWD_DRAW_FULL = 0,
    CROWD_DRAW_SKIN_DISCARD,
    CROWD_DRAW_STATIC_DISCARD
} CrowdDrawMode;

/* The skinned mesh GPU buffers, shared by every tier (baked GPU crowd, CPU heroes, CPU-batch
 * crowd). Extracted from Crowd so the CPU tier can run on a rig with no BAKED section - Crowd
 * requires BAKED, this does not. Owned by main; passed in. */
typedef struct {
    VkBuffer vbuf, ibuf; VkDeviceMemory vmem, imem;
    uint32_t index_count, vert_count;
    int      cull_back;   /* winding is CCW-outward everywhere -> safe to back-face cull */
} SharedMesh;

int  shared_mesh_init(SharedMesh *m, VkCtx *ctx, const ProcAssets *assets);
void shared_mesh_destroy(SharedMesh *m, VkCtx *ctx);

typedef struct {
    /* baked palette texture (bindless) */
    VkImage palette_img; VkDeviceMemory palette_mem; VkImageView palette_view;
    VkSampler sampler;
    VkDescriptorPool desc_pool; VkDescriptorSetLayout set_layout; VkDescriptorSet set;
    VkPipelineLayout layout;
    VkShaderEXT vs, fs;
    VkShaderEXT vs_skin_bench, vs_static_bench;   /* bench-only vertex-skinning microbench VS variants */
    VkShaderEXT vs_skel, fs_skel;                 /* bone-line skeleton render LOD (LINE_LIST) */

    /* Static bone-line geometry: two endpoints per non-root joint, each {joint, bind-model origin}.
     * Shared across all instances via the instanced draw; posed on the GPU from the baked palette. */
    VkBuffer skel_vbuf; VkDeviceMemory skel_vmem;
    uint32_t skel_vert_count;

    const SharedMesh *mesh;   /* borrowed; owned by main */

    /* instances - one buffer copy per frame-in-flight so the CPU never overwrites a buffer the
     * GPU is still reading from an earlier frame (the begin_frame throttle frees the slot first).
     * Buffers + the cpu mirror are sized to `capacity`; only the first `count` are drawn/updated,
     * so the live entity count is adjustable (crowd_set_count) with no reallocation. */
    uint32_t capacity;
    uint32_t count;
    InstanceAnim *cpu;
    VkBuffer inst_buf[VKC_FRAMES_IN_FLIGHT]; VkDeviceMemory inst_mem[VKC_FRAMES_IN_FLIGHT];
    void *inst_map[VKC_FRAMES_IN_FLIGHT];    VkDeviceAddress inst_addr[VKC_FRAMES_IN_FLIGHT];

    CrowdClip clips[DEMO_PROC_MAX_CLIPS];
    uint32_t clip_count;
} Crowd;

/* Allocate buffers for `capacity` instances; start with `count` (<= capacity) drawn. */
int  crowd_init(Crowd *cr, VkCtx *ctx, const ProcAssets *assets, const SharedMesh *mesh,
                uint32_t capacity, uint32_t count);
/* Change the live instance count (clamped to [1, capacity]) and rebuild the grid layout. Touches
 * only the CPU mirror (re-uploaded per frame by crowd_update), so no GPU buffer is in flight. */
void crowd_set_count(Crowd *cr, uint32_t count);
/* Advance the animation state and stage it into this frame-in-flight's instance buffer (`frame`
 * = ctx->cur_frame, valid only after vkc_begin_frame has throttled that slot). */
void crowd_update(Crowd *cr, float dt, uint32_t frame);
void crowd_draw(Crowd *cr, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent,
                CrowdDrawMode mode);
/* Draw `count` instances from an arbitrary InstanceAnim buffer (device address) through the baked
 * GPU shader + bindless palette + shared mesh. Used both for the crowd and to render the heroes via
 * the baked path for the LOD-promotion A/B toggle. `mode` selects the real render or a discard bench VS. */
void crowd_draw_instances(Crowd *cr, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj,
                          VkExtent2D extent, VkDeviceAddress inst_addr, uint32_t count,
                          CrowdDrawMode mode);
/* Draw `count` instances as bone-line skeletons from an arbitrary InstanceAnim buffer: a LINE_LIST
 * over the static bone geometry, each endpoint posed by its joint's baked palette row. The cheap
 * render LOD - per-vertex skinning cost drops to ~2x bones while the animation cost is unchanged. */
void crowd_draw_skeleton(Crowd *cr, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj,
                         VkExtent2D extent, VkDeviceAddress inst_addr, uint32_t count);
void crowd_destroy(Crowd *cr, VkCtx *ctx);

#endif /* DEMO_CROWD_H */
