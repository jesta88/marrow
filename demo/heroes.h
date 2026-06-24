/* CPU-animated hero characters: the CPU pose pipeline + LOD-promotion counterpart to the baked GPU crowd.
 *
 * Each frame, for every hero, marrow samples the dense clips, blends them with the pose
 * primitive (mrw_pose_blend - the hemisphere-corrected nlerp cross-fade), composes the hierarchy and
 * the skinning palette (mrw_local_to_model + mrw_model_to_palette), and uploads the canonical 3x4
 * affines to a buffer-device-address palette buffer. Root motion (mrw_root_motion) walks the heroes
 * across the ground. The skin_tierA.vert shader fetches the palette and does linear-blend skinning
 * with the GENERAL cofactor normal. */
#ifndef DEMO_HEROES_H
#define DEMO_HEROES_H

#include "vk_context.h"
#include "assets_proc.h"
#include "linalg.h"
#include "marrow.h"
#include "crowd.h"     /* InstanceAnim, crowd_draw_instances - for the baked-path promotion A/B */

typedef struct {
    float model[16];
    float tint[4];
} HeroInstance;   /* mirrors GLSL HeroInstance (scalar layout); 80 bytes */

typedef struct {
    float t, prev_t;     /* playback time (monotonic; marrow wraps for looping) */
    float w;             /* cross-fade weight walk->run */
    float wphase;        /* phase for the oscillating w  */
    float yaw;
    float pos[3];
    float tint[4];
} HeroState;

typedef struct {
    VkShaderEXT vs, fs; VkPipelineLayout layout;
    /* one copy per frame-in-flight: each frame writes only its own slot, so the CPU never clobbers
     * buffers the GPU is still reading from an earlier in-flight frame (see crowd.h). */
    VkBuffer hero_buf[VKC_FRAMES_IN_FLIGHT]; VkDeviceMemory hero_mem[VKC_FRAMES_IN_FLIGHT];
    void *hero_map[VKC_FRAMES_IN_FLIGHT];    VkDeviceAddress hero_addr[VKC_FRAMES_IN_FLIGHT];
    VkBuffer pal_buf[VKC_FRAMES_IN_FLIGHT];  VkDeviceMemory pal_mem[VKC_FRAMES_IN_FLIGHT];
    void *pal_map[VKC_FRAMES_IN_FLIGHT];     VkDeviceAddress pal_addr[VKC_FRAMES_IN_FLIGHT];

    /* parallel baked-path representation of the same heroes (same models/clip/phase) for the A/B toggle */
    VkBuffer tierb_buf[VKC_FRAMES_IN_FLIGHT]; VkDeviceMemory tierb_mem[VKC_FRAMES_IN_FLIGHT];
    void *tierb_map[VKC_FRAMES_IN_FLIGHT];    VkDeviceAddress tierb_addr[VKC_FRAMES_IN_FLIGHT];
    uint32_t wb_first, wb_count, wb_loop, rb_first, rb_count, rb_loop;

    uint32_t count, joint_count;

    /* marrow views borrow assets->blob, which must stay alive for the Heroes lifetime */
    mrw_blob blob;
    mrw_skeleton_view skel;
    mrw_clip_view walk, run;
    float walk_dur, run_dur;

    HeroState *state;
    HeroInstance *cpu;

    /* CPU scratch for the pose pipeline */
    mrw_trs *localsA, *localsB;
    float   *model12;

    /* CPU pose showcase (opt-in via --showcase): an additive upper-body lean (mrw_pose_accumulate
     * + a per-joint mask), a head look-at (mrw_ik_aim), and two-bone foot IK (mrw_ik_two_bone) planting
     * the feet on the walked ground. Built on the procedural biped's named joints; auto-disabled on a
     * rig that lacks them. Off by default so the headless --validate/--screenshot stay bit-identical. */
    int      showcase_ok;                       /* the named joints were all resolved */
    int      j_head, j_lt, j_ls, j_lf, j_rt, j_rs, j_rf;   /* head + L/R thigh/shin/foot */
    float   *ub_mask;                           /* joint_count upper-body mask (1 above the pelvis) */
    mrw_trs *lean_delta;                        /* joint_count additive lean (forward pitch) */
    float    foot_y[2];                         /* L/R rest foot model-space ground height */
} Heroes;

int  heroes_init(Heroes *h, VkCtx *ctx, const ProcAssets *assets, uint32_t count);
/* Enable the CPU pose showcase on subsequently-updated heroes (process-global; --showcase). */
void heroes_set_showcase(int on);
/* Run the CPU pose pipeline and stage the palette / instance data into this frame-in-flight's
 * buffers (`frame` = ctx->cur_frame, valid only after vkc_begin_frame has throttled that slot). */
void heroes_update(Heroes *h, float dt, uint32_t frame);
/* Heroes reuse the shared skinned mesh (owned by the Crowd); pass its buffers + index count. */
void heroes_draw(Heroes *h, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent,
                 VkBuffer vbuf, VkBuffer ibuf, uint32_t index_count);
void heroes_destroy(Heroes *h, VkCtx *ctx);

#endif /* DEMO_HEROES_H */
