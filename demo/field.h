/* Unified view-LOD crowd field - the demo's headline scene.
 *
 * One entity field. Each frame every entity is classified by distance to the camera into two
 * INDEPENDENT LODs:
 *   - ANIMATION tier (radius r_a): near -> Tier A (CPU, mrw_batch_blend_clips_to_palette, exact
 *     local-space clip cross-fade); far -> Tier B (baked GPU palette, crowd.vert).
 *   - RENDER LOD (radius r_mesh): near -> full skinned mesh; far -> bone-line skeleton (a cheap proxy
 *     that collapses ~432 verts/instance to ~2x bones, keeping the far tail off the vertex wall). A
 *     global toggle (skeleton_all) renders the whole field as lines for the raw-scale view.
 *
 * Near and far run the SAME animation, differing only by tier - marrow's LOD-promotion contract made
 * visible: a foreground of CPU Tier-A characters seamlessly continuing into a massive baked Tier-B
 * crowd. Tier B has no instance ceiling, so the field scales to tens of thousands.
 *
 * The field OWNS the canonical entity array, the persistent per-entity LOD state, the near Tier-A
 * buffers/scratch (its OWN, sized to near_cap - deliberately NOT CrowdCpu, which is hard-capped at
 * CROWD_CPU_MAX), and the far Tier-B InstanceAnim buffers (sized to total_capacity). It BORROWS a
 * Crowd purely for the Tier-B draw machinery (bindless baked palette + pipeline + crowd_draw_instances)
 * and the shared skinned mesh. */
#ifndef DEMO_FIELD_H
#define DEMO_FIELD_H

#include "vk_context.h"
#include "assets_proc.h"
#include "linalg.h"
#include "marrow.h"
#include "crowd.h"       /* Crowd (borrowed Tier-B), InstanceAnim, SharedMesh */
#include "profiler.h"

/* Per-entity record (capacity-sized, allocated once at init): a fixed grid position, the two clips it
 * cross-fades, its animation phase/weight, and a STABLE id. The id drives the per-instance tint in
 * BOTH tiers (so an entity keeps one color as it promotes/demotes) and breaks distance ties in the
 * near-set top-K (so the selection is deterministic). */
typedef struct {
    float    pos[3];
    uint16_t clipA, clipB;   /* clip-table indices (the cross-fade pair)               */
    float    phase, w;       /* playback time; blend weight clipA<->clipB              */
    uint32_t id;             /* stable entity id (tint key + top-K tie-break)          */
} FieldEntity;

/* Distance thresholds (world units). Each split carries a hysteresis band (enter inside the radius,
 * exit only past 1.1x it) so borderline entities don't flip tier every frame. */
typedef struct {
    float r_a;       /* animation-tier radius: nearer -> Tier A (CPU)          */
    float r_mesh;    /* render-LOD radius: nearer -> full mesh                  */
} FieldLod;

/* A contiguous run of near entities sharing one (clipA,clipB) pair - one homogeneous batched-blend
 * call. The near set is counting-sorted into these so the kernel runs once per pair. */
typedef struct { uint16_t clipA, clipB; uint32_t start, count; } FieldGroup;

#define FIELD_MAX_GROUPS (DEMO_PROC_MAX_CLIPS * DEMO_PROC_MAX_CLIPS)

typedef struct {
    /* borrowed (owned by the caller) */
    Crowd            *crowd;        /* Tier-B draw machinery; the shared mesh is crowd->mesh */
    uint32_t          joint_count;

    /* clip tables resolved from the blob (the views borrow it; it stays alive for the field's life) */
    mrw_blob          blob;
    mrw_skeleton_view skel;
    mrw_clip_view     clipv[DEMO_PROC_MAX_CLIPS];        /* Tier A dense clips                 */
    uint32_t          baked_first[DEMO_PROC_MAX_CLIPS];  /* Tier B baked clip-table entries    */
    uint32_t          baked_count[DEMO_PROC_MAX_CLIPS];
    uint32_t          baked_loop[DEMO_PROC_MAX_CLIPS];
    float             clip_dur[DEMO_PROC_MAX_CLIPS];
    uint32_t          clip_count;

    /* canonical entity field + persistent LOD state (kept in separate arrays so the hot partition
     * scan stays SoA-friendly; the LOD bytes hold last frame's decision, which the hysteresis reads) */
    FieldEntity      *ent;
    uint8_t          *anim_tier;   /* 0 = Tier A, 1 = Tier B                                 */
    uint8_t          *render_lod;  /* 0 = mesh,   1 = skeleton                               */
    uint32_t          total_capacity;
    uint32_t          count;       /* live entity count                                      */
    FieldLod          lod;

    /* per-frame partition scratch (capacity-sized, allocated once) */
    void             *cand;        /* near candidates for the top-K select (private struct)  */
    uint8_t          *near_mark;   /* 1 = this entity is in the final near set this frame     */

    /* near Tier-A set - OWN buffers/scratch, sized to near_cap (independent of CrowdCpu's ceiling) */
    uint32_t          near_cap;
    float            *timesA, *timesB, *weights;  /* contiguous per-near-instance batch inputs */
    void             *hero_stage;                 /* near_cap model+tint, cacheable (private struct) */
    float            *pal_scratch;  size_t pal_bytes;    /* 64-aligned cacheable batched-blend output */
    void             *batch_scratch; size_t batch_bytes; /* 64-aligned kernel SoA tile               */
    FieldGroup        groups[FIELD_MAX_GROUPS]; uint32_t group_count;
    mrw_dispatch      disp;            /* near batched-blend backend (best the host supports)   */
    const char       *backend_name;

    /* per-frame-in-flight near GPU buffers (model+tint upload + f32 palette SSBO the shader fetches) */
    VkBuffer hero_buf[VKC_FRAMES_IN_FLIGHT]; VkDeviceMemory hero_mem[VKC_FRAMES_IN_FLIGHT];
    void *hero_map[VKC_FRAMES_IN_FLIGHT];    VkDeviceAddress hero_addr[VKC_FRAMES_IN_FLIGHT];
    VkBuffer pal_buf[VKC_FRAMES_IN_FLIGHT];  VkDeviceMemory pal_mem[VKC_FRAMES_IN_FLIGHT];
    void *pal_map[VKC_FRAMES_IN_FLIGHT];     VkDeviceAddress pal_addr[VKC_FRAMES_IN_FLIGHT];

    /* near pipeline (skin_tierA.vert + crowd.frag, its own layout - same scalar push as the heroes).
     * vs_skel/fs_skel are the bone-line variant (skin_tierA_skel.vert + skel.frag) sharing that layout
     * and the borrowed crowd's static bone VB; used only under the global all-skeleton toggle. */
    VkShaderEXT vs, fs; VkPipelineLayout layout;
    VkShaderEXT vs_skel, fs_skel;

    /* far Tier-B set - OWN InstanceAnim buffers, sized to total_capacity (Tier B has no 16k cap). The
     * far set is compacted into two contiguous sub-ranges: the full-mesh band [0, far_mesh_count) then
     * the bone-line skeleton tail [far_mesh_count, far_count). */
    InstanceAnim     *far_stage;       /* total_capacity, cacheable; bulk-copied to the WC buffer */
    VkBuffer far_buf[VKC_FRAMES_IN_FLIGHT]; VkDeviceMemory far_mem[VKC_FRAMES_IN_FLIGHT];
    void *far_map[VKC_FRAMES_IN_FLIGHT];    VkDeviceAddress far_addr[VKC_FRAMES_IN_FLIGHT];

    /* global render-mode toggle (--skeleton / K): draw the WHOLE field (near + far) as bone lines.
     * When set, field_update forces every far entity into the skeleton sub-range and the near set is
     * drawn through skin_tierA_skel.vert instead of the full mesh. */
    int skeleton_all;

    /* this frame's split (published for the HUD / counters) */
    uint32_t near_count, far_count, near_clamped;
    uint32_t far_mesh_count, far_skel_count;   /* far_count = far_mesh_count + far_skel_count */
} Field;

/* Build the field for `count` live entities (clamped to total_capacity). The borrowed Crowd supplies
 * the Tier-B draw path + shared mesh and MUST have a BAKED section. Returns 0 (nothing leaked) on
 * failure. */
int  field_init(Field *f, VkCtx *ctx, const ProcAssets *assets, Crowd *crowd,
                uint32_t total_capacity, uint32_t near_cap, uint32_t count, FieldLod lod);
/* Change the live entity count (clamped to [1, total_capacity]) and rebuild the grid + reset the LOD
 * state. CPU-only (the per-frame buffers are re-staged by field_update), so no device idle needed. */
void field_set_count(Field *f, uint32_t count);
/* Advance phases, partition by distance (PROF_LOD), run the near batched blend (PALETTE_GEN), and
 * stage near + far into this frame-in-flight's buffers (PALETTE_UPLOAD). `frame` = ctx->cur_frame. */
void field_update(Field *f, const float cam_pos[3], float dt, uint32_t frame, Profiler *prof);
/* Far Tier-B full-mesh band [0, far_mesh_count): one instanced baked-palette mesh draw. */
void field_draw_far(Field *f, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent);
/* Far Tier-B skeleton tail [far_mesh_count, far_count): one instanced bone-line draw past R_mesh. */
void field_draw_far_skeleton(Field *f, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent);
/* Near Tier-A band: one instanced skinned-mesh draw fetching the CPU-computed palette. Skipped under
 * the global all-skeleton toggle (the near set then draws through field_draw_near_skeleton instead). */
void field_draw_near(Field *f, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent);
/* Near Tier-A bone lines: the near set drawn as a skeleton from its CPU palette - only under the
 * global all-skeleton toggle (a no-op otherwise). */
void field_draw_near_skeleton(Field *f, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent);
void field_destroy(Field *f, VkCtx *ctx);

#endif /* DEMO_FIELD_H */
