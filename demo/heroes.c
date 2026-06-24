#include "heroes.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

_Static_assert(sizeof(HeroInstance) == 80, "HeroInstance must be 80 bytes");
_Static_assert(offsetof(HeroInstance, model) ==  0, "HeroInstance.model");
_Static_assert(offsetof(HeroInstance, tint)  == 64, "HeroInstance.tint");

typedef struct {
    float           viewProj[16];
    VkDeviceAddress heroes;
    VkDeviceAddress palettes;
    uint32_t        jointCount;
    uint32_t        _pad;
} HeroPush;
_Static_assert(sizeof(HeroPush) == 88, "HeroPush must be 88 bytes");
_Static_assert(offsetof(HeroPush, viewProj)   ==  0, "HeroPush.viewProj");
_Static_assert(offsetof(HeroPush, heroes)     == 64, "HeroPush.heroes");
_Static_assert(offsetof(HeroPush, palettes)   == 72, "HeroPush.palettes");
_Static_assert(offsetof(HeroPush, jointCount) == 80, "HeroPush.jointCount");

#include "skin_tierA_vert.spv.h"
#include "crowd_frag.spv.h"   /* reuse the crowd fragment (same vNormal/vColor inputs) */

/* --showcase (process-global; all heroes share the mode). Off ⇒ the default pose path is unchanged,
 * so the headless --validate/--screenshot stay bit-identical. */
static int g_showcase = 0;
void heroes_set_showcase(int on) { g_showcase = on; }

/* resolve a joint by name (the procedural biped names its bones); -1 if absent. */
static int find_joint(const mrw_skeleton_view *skel, const char *name) {
    for (uint32_t j = 0; j < skel->joint_count; ++j) {
        const char *n = NULL;
        if (mrw_skeleton_joint_name(skel, j, &n) == MRW_OK && n && strcmp(n, name) == 0) return (int)j;
    }
    return -1;
}
static void axis_angle(int axis, float ang, float q[4]) {
    float h = ang * 0.5f; q[0] = q[1] = q[2] = 0.0f; q[3] = cosf(h); q[axis] = sinf(h);
}

/* find the section index of the clip named `name` (section = 1 + clip_index) */
static int find_clip(const ProcAssets *a, const char *name, mrw_blob *blob, mrw_clip_view *out,
                     float *dur, uint32_t *clip_index) {
    for (uint32_t i = 0; i < a->clip_count; ++i) {
        if (strcmp(a->clips[i].name, name) == 0) {
            if (mrw_clip_view_at(blob, 1 + a->clips[i].clip_index, out) != MRW_OK) return 0;
            *dur = a->clips[i].duration_s;
            *clip_index = a->clips[i].clip_index;
            return 1;
        }
    }
    return 0;
}

static int find_baked(const mrw_blob *b, mrw_baked_view *out) {
    for (uint32_t i = 0; i < b->section_count; ++i) {
        uint32_t type = 0;
        if (mrw_blob_section_type(b, i, &type) == MRW_OK && type == MRW_SECTION_BAKED)
            return mrw_baked_view_at(b, i, out) == MRW_OK;
    }
    return 0;
}

int heroes_init(Heroes *h, VkCtx *ctx, const ProcAssets *assets, uint32_t count) {
    memset(h, 0, sizeof *h);
    h->count = count;
    h->joint_count = assets->joint_count;

    if (mrw_blob_open(assets->blob, assets->blob_size, &h->blob) != MRW_OK) { fprintf(stderr, "[heroes] blob open\n"); return 0; }
    mrw_blob_skeleton(&h->blob, &h->skel);
    uint32_t walk_ci, run_ci;
    if (!find_clip(assets, "walk", &h->blob, &h->walk, &h->walk_dur, &walk_ci)) { fprintf(stderr, "[heroes] no walk clip\n"); return 0; }
    if (!find_clip(assets, "run",  &h->blob, &h->run,  &h->run_dur,  &run_ci))  { fprintf(stderr, "[heroes] no run clip\n"); return 0; }

    /* baked clip-table entries for the baked-path A/B representation */
    mrw_baked_view bv;
    if (find_baked(&h->blob, &bv)) {
        mrw_baked_clip we, re;
        mrw_baked_clip_entry(&bv, walk_ci, &we);
        mrw_baked_clip_entry(&bv, run_ci, &re);
        h->wb_first = we.first_frame; h->wb_count = we.frame_count; h->wb_loop = (we.flags & MRW_BAKED_CLIP_LOOPING) ? 1u : 0u;
        h->rb_first = re.first_frame; h->rb_count = re.frame_count; h->rb_loop = (re.flags & MRW_BAKED_CLIP_LOOPING) ? 1u : 0u;
    }

    uint32_t jc = h->joint_count;
    h->localsA = (mrw_trs *)malloc((size_t)jc * sizeof(mrw_trs));
    h->localsB = (mrw_trs *)malloc((size_t)jc * sizeof(mrw_trs));
    h->model12 = (float *)malloc((size_t)jc * 12 * sizeof(float));
    h->state = (HeroState *)calloc(count, sizeof(HeroState));
    h->cpu = (HeroInstance *)calloc(count, sizeof(HeroInstance));

    /* showcase data (built unconditionally - cheap; applied only when --showcase is set). */
    h->ub_mask    = (float *)malloc((size_t)jc * sizeof(float));
    h->lean_delta = (mrw_trs *)malloc((size_t)jc * sizeof(mrw_trs));
    h->j_head = find_joint(&h->skel, "head");
    h->j_lt = find_joint(&h->skel, "L_thigh"); h->j_ls = find_joint(&h->skel, "L_shin"); h->j_lf = find_joint(&h->skel, "L_foot");
    h->j_rt = find_joint(&h->skel, "R_thigh"); h->j_rs = find_joint(&h->skel, "R_shin"); h->j_rf = find_joint(&h->skel, "R_foot");
    /* the showcase needs both scratch buffers AND the named chain joints - else it stays disabled */
    h->showcase_ok = (h->ub_mask && h->lean_delta &&
                      h->j_head>=0 && h->j_lt>=0 && h->j_ls>=0 && h->j_lf>=0 && h->j_rt>=0 && h->j_rs>=0 && h->j_rf>=0);
    if (h->ub_mask && h->lean_delta) {
        /* upper-body mask: 1 above the pelvis; 0 on the pelvis + legs/feet (the lean leaves them planted) */
        for (uint32_t j = 0; j < jc; ++j) {
            const char *n = NULL; mrw_skeleton_joint_name(&h->skel, j, &n);
            int lower = !n || strstr(n, "thigh") || strstr(n, "shin") || strstr(n, "foot") || strcmp(n, "pelvis") == 0;
            h->ub_mask[j] = lower ? 0.0f : 1.0f;
        }
        /* lean delta: a uniform forward pitch on every joint - the mask restricts it to the upper body */
        float lq[4]; axis_angle(0, 0.20f, lq);
        for (uint32_t j = 0; j < jc; ++j) {
            for (int k = 0; k < 4; ++k) h->lean_delta[j].rot[k] = lq[k];
            for (int k = 0; k < 3; ++k) { h->lean_delta[j].trans[k] = 0.0f; h->lean_delta[j].scale[k] = 1.0f; }
        }
    }
    /* rest-pose foot ground heights (model space) for the planting IK */
    if (h->showcase_ok) {
        mrw_trs *rl = (mrw_trs *)malloc((size_t)jc * sizeof(mrw_trs));
        for (uint32_t j = 0; j < jc; ++j) mrw_skeleton_rest_local(&h->skel, j, &rl[j]);
        mrw_local_to_model(&h->skel, rl, h->model12, jc);
        h->foot_y[0] = h->model12[h->j_lf*12+7];
        h->foot_y[1] = h->model12[h->j_rf*12+7];
        free(rl);
    }

    /* lay heroes out in a row in front of the crowd; vivid gold tint to read as the CPU-animated set */
    for (uint32_t i = 0; i < count; ++i) {
        HeroState *s = &h->state[i];
        s->pos[0] = ((float)i - 0.5f * (float)(count - 1)) * 2.2f;
        s->pos[1] = 0.0f;
        s->pos[2] = 20.0f;
        s->yaw = 0.0f;                              /* face -Z, walk away from camera */
        s->t = (float)i * 0.21f;
        s->prev_t = s->t;
        s->wphase = (float)i * 0.5f;
        float g = 0.92f + 0.08f * sinf((float)i);
        s->tint[0] = g; s->tint[1] = 0.80f * g; s->tint[2] = 0.12f; s->tint[3] = 1.0f;  /* bright gold */
    }

    /* GPU buffers - one copy per frame-in-flight: per-instance state + CPU-computed palette +
     * the parallel baked-path entries (all BDA storage buffers) */
    VkDeviceSize hero_bytes  = (VkDeviceSize)count * sizeof(HeroInstance);
    VkDeviceSize pal_bytes   = (VkDeviceSize)count * jc * 12 * sizeof(float);
    VkDeviceSize tierb_bytes = (VkDeviceSize)count * sizeof(InstanceAnim);
    VkBufferUsageFlags su = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    for (uint32_t f = 0; f < VKC_FRAMES_IN_FLIGHT; ++f) {
        if (!vkc_create_host_buffer(ctx, hero_bytes,  su, &h->hero_buf[f],  &h->hero_mem[f],  &h->hero_map[f]))  return 0;
        if (!vkc_create_host_buffer(ctx, pal_bytes,   su, &h->pal_buf[f],   &h->pal_mem[f],   &h->pal_map[f]))   return 0;
        if (!vkc_create_host_buffer(ctx, tierb_bytes, su, &h->tierb_buf[f], &h->tierb_mem[f], &h->tierb_map[f])) return 0;
        h->hero_addr[f]  = vkc_buffer_address(ctx, h->hero_buf[f]);
        h->pal_addr[f]   = vkc_buffer_address(ctx, h->pal_buf[f]);
        h->tierb_addr[f] = vkc_buffer_address(ctx, h->tierb_buf[f]);
    }

    VkPushConstantRange pr = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(HeroPush) };
    VkPipelineLayoutCreateInfo li = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(ctx->device, &li, NULL, &h->layout) != VK_SUCCESS) return 0;

    if (!vkc_create_graphics_shaders(ctx, skin_tierA_vert_spv, sizeof skin_tierA_vert_spv,
            crowd_frag_spv, sizeof crowd_frag_spv, NULL, 0, &pr, &h->vs, &h->fs)) return 0;

    fprintf(stderr, "[heroes] %u Tier-A heroes (CPU pose pipeline + root motion)\n", count);
    return 1;
}

void heroes_update(Heroes *h, float dt, uint32_t frame) {
    uint32_t jc = h->joint_count;
    float *pal = (float *)h->pal_map[frame];

    for (uint32_t i = 0; i < h->count; ++i) {
        HeroState *s = &h->state[i];
        s->prev_t = s->t;
        s->t += dt;
        s->wphase += dt * 0.6f;
        s->w = 0.5f + 0.5f * sinf(s->wphase);   /* continuous walk<->run cross-fade */

        /* sample both clips, blend (marrow's pose primitive, in place), compose hierarchy + palette */
        mrw_clip_sample_local(&h->walk, s->t, h->localsA, jc);
        mrw_clip_sample_local(&h->run,  s->t, h->localsB, jc);
        mrw_pose_blend(h->localsA, h->localsB, s->w, NULL, h->localsA, jc, jc);

        if (g_showcase && h->showcase_ok) {
            /* (a) additive upper-body lean: mrw_pose_accumulate of the lean delta with the upper-body
               mask, at an oscillating weight (legs stay put - masked out). */
            float leanw = 0.4f + 0.4f * sinf(s->t * 1.2f + (float)i);
            mrw_pose_accumulate(h->localsA, h->lean_delta, leanw, h->ub_mask, h->localsA, jc, jc);
            mrw_local_to_model(&h->skel, h->localsA, h->model12, jc);
            /* (b) head look-at: aim the head's local −Z at a point swinging in front (mrw_ik_aim). */
            float aim_axis[3] = { 0, 0, -1 }, up_axis[3] = { 0, 1, 0 }, up_model[3] = { 0, 1, 0 };
            float tgt[3] = { 0.7f * sinf(s->t * 0.8f + (float)i), h->model12[h->j_head*12+7] + 0.1f, -1.2f };
            mrw_ik_aim(&h->skel, h->model12, h->localsA, (uint32_t)h->j_head,
                       aim_axis, up_axis, tgt, up_model, 0.85f, jc);
            /* (c) two-bone foot IK: plant each foot at its animated x/z but pinned to the rest ground
               height, knee poled forward (mrw_ik_two_bone). Head + both legs are disjoint subtrees, so
               all three solve against this model before one final recompose. */
            int legs[2][3] = { { h->j_lt, h->j_ls, h->j_lf }, { h->j_rt, h->j_rs, h->j_rf } };
            for (int L = 0; L < 2; ++L) {
                const float *fm = &h->model12[legs[L][2]*12];
                const float *km = &h->model12[legs[L][1]*12];
                float ft[3]   = { fm[3], h->foot_y[L], fm[11] };
                float pole[3] = { km[3], km[7], km[11] - 1.0f };
                mrw_ik_two_bone(&h->skel, h->model12, h->localsA,
                                (uint32_t)legs[L][0], (uint32_t)legs[L][1], (uint32_t)legs[L][2],
                                ft, pole, 1.0f, jc);
            }
            mrw_local_to_model(&h->skel, h->localsA, h->model12, jc);   /* recompose (lean + aim + IK) */
        } else {
            mrw_local_to_model(&h->skel, h->localsA, h->model12, jc);
        }
        mrw_model_to_palette(&h->skel, h->model12, &pal[(size_t)i * jc * 12], jc);

        /* root motion: forward travel, faster while "running" (higher w) */
        mrw_xform delta;
        mrw_root_motion(&h->walk, s->prev_t, s->t, &delta);
        float c = cosf(s->yaw), sn = sinf(s->yaw);
        float scale = 1.0f + s->w;                 /* run blend moves faster */
        float dx = delta.trans[0] * scale, dz = delta.trans[2] * scale;
        s->pos[0] += c * dx + sn * dz;
        s->pos[2] += -sn * dx + c * dz;
        if (s->pos[2] < -4.0f) s->pos[2] += 26.0f; /* wrap back to the front */

        /* heroes are rendered larger (uniform scale) so they read as the foreground CPU-animated set */
        mat4 model = mat4_mul(mat4_mul(mat4_translate(v3(s->pos[0], s->pos[1], s->pos[2])),
                                       mat4_rotate_y(s->yaw)), mat4_scale(v3(1.8f, 1.8f, 1.8f)));
        memcpy(h->cpu[i].model, model.m, sizeof model.m);
        memcpy(h->cpu[i].tint, s->tint, sizeof s->tint);

        /* parallel baked-path entry: same model + the SAME walk<->run cross-fade, but evaluated by the
         * baked component-space path (the honest LOD-promotion A/B; differs by the cross-fade gap). */
        InstanceAnim *tb = &((InstanceAnim *)h->tierb_map[frame])[i];
        memcpy(tb->model, model.m, sizeof model.m);
        tb->clipA[0] = h->wb_first; tb->clipA[1] = h->wb_count; tb->clipA[2] = h->wb_loop; tb->clipA[3] = 0;
        tb->clipB[0] = h->rb_first; tb->clipB[1] = h->rb_count; tb->clipB[2] = h->rb_loop; tb->clipB[3] = 0;
        tb->times[0] = s->t; tb->times[1] = s->t; tb->times[2] = h->walk_dur; tb->times[3] = h->run_dur;
        tb->blend[0] = s->w; tb->blend[1] = tb->blend[2] = tb->blend[3] = 0.0f;
    }
    memcpy(h->hero_map[frame], h->cpu, (size_t)h->count * sizeof(HeroInstance));
}

void heroes_draw(Heroes *h, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent,
                 VkBuffer vbuf, VkBuffer ibuf, uint32_t index_count) {
    VkVertexInputBindingDescription2EXT b = { VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT };
    b.binding = 0; b.stride = sizeof(DemoVertex); b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; b.divisor = 1;
    VkVertexInputAttributeDescription2EXT a[5];
    for (int i = 0; i < 5; ++i) { a[i] = (VkVertexInputAttributeDescription2EXT){ VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT }; a[i].binding = 0; a[i].location = (uint32_t)i; }
    a[0].format = VK_FORMAT_R32G32B32_SFLOAT;    a[0].offset = offsetof(DemoVertex, pos);
    a[1].format = VK_FORMAT_R32G32B32_SFLOAT;    a[1].offset = offsetof(DemoVertex, nrm);
    a[2].format = VK_FORMAT_R32G32B32A32_SFLOAT; a[2].offset = offsetof(DemoVertex, tan);
    a[3].format = VK_FORMAT_R32G32B32A32_UINT;   a[3].offset = offsetof(DemoVertex, bones);
    a[4].format = VK_FORMAT_R32G32B32A32_SFLOAT; a[4].offset = offsetof(DemoVertex, weights);

    vkc_bind_shaders(ctx, cmd, h->vs, h->fs);
    vkc_set_default_state(ctx, cmd, extent, &b, 1, a, 5);
    vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);

    HeroPush pc;
    memcpy(pc.viewProj, view_proj->m, sizeof pc.viewProj);
    pc.heroes = h->hero_addr[ctx->cur_frame];
    pc.palettes = h->pal_addr[ctx->cur_frame];
    pc.jointCount = h->joint_count;
    pc._pad = 0;
    vkCmdPushConstants(cmd, h->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof pc, &pc);

    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf, &zero);
    vkCmdBindIndexBuffer(cmd, ibuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, index_count, h->count, 0, 0, 0);
}

void heroes_destroy(Heroes *h, VkCtx *ctx) {
    vkDeviceWaitIdle(ctx->device);
    vkc_destroy_shader(ctx, h->vs); vkc_destroy_shader(ctx, h->fs);
    if (h->layout) vkDestroyPipelineLayout(ctx->device, h->layout, NULL);
    for (uint32_t f = 0; f < VKC_FRAMES_IN_FLIGHT; ++f) {
        vkc_destroy_buffer(ctx, h->hero_buf[f],  h->hero_mem[f]);
        vkc_destroy_buffer(ctx, h->pal_buf[f],   h->pal_mem[f]);
        vkc_destroy_buffer(ctx, h->tierb_buf[f], h->tierb_mem[f]);
    }
    free(h->state); free(h->cpu); free(h->localsA); free(h->localsB); free(h->model12);
    free(h->ub_mask); free(h->lean_delta);
    memset(h, 0, sizeof *h);
}
