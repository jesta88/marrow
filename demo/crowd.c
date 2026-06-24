#include "crowd.h"

#include "marrow.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#define MAX_PALETTES 16

/* GPU-mirrored struct + push constant - must match crowd.vert (scalar layout). */
_Static_assert(sizeof(InstanceAnim) == 128, "InstanceAnim must be 128 bytes");
_Static_assert(offsetof(InstanceAnim, clipA) == 64, "clipA offset");
_Static_assert(offsetof(InstanceAnim, clipB) == 80, "clipB offset");
_Static_assert(offsetof(InstanceAnim, times) == 96, "times offset");
_Static_assert(offsetof(InstanceAnim, blend) == 112, "blend offset");

typedef struct {
    float           viewProj[16];
    VkDeviceAddress instances;
    uint32_t        texelsPerBone;
    float           benchEps;   /* bench-only: nonzero in the rasterizer-discard skin/static modes so
                                   the skinning is folded into gl_Position and can't be DCE'd; 0 otherwise */
} CrowdPush;
_Static_assert(sizeof(CrowdPush) == 80, "CrowdPush must be 80 bytes");

/* Generated SPIR-V */
#include "crowd_vert.spv.h"
#include "crowd_frag.spv.h"
#include "crowd_skin_bench_vert.spv.h"      /* crowd.vert built with -DMARROW_BENCH */
#include "crowd_static_bench_vert.spv.h"
#include "crowd_skel_vert.spv.h"            /* bone-line skeleton render LOD */
#include "skel_frag.spv.h"

/* One endpoint of a bone line: the joint whose baked palette row poses it, and the joint's
 * bind-model-space origin. Posed on the GPU as M_joint . origin (crowd_skel.vert). 16 bytes. */
typedef struct { uint32_t joint; float origin[3]; } SkelVertex;

static int find_baked(const mrw_blob *b, mrw_baked_view *out) {
    for (uint32_t i = 0; i < b->section_count; ++i) {
        uint32_t type = 0;
        if (mrw_blob_section_type(b, i, &type) == MRW_OK && type == MRW_SECTION_BAKED)
            return mrw_baked_view_at(b, i, out) == MRW_OK;
    }
    return 0;
}

/* ------------------------------------------------------------------ shared skinned mesh */

int shared_mesh_init(SharedMesh *m, VkCtx *ctx, const ProcAssets *assets) {
    memset(m, 0, sizeof *m);
    void *p;
    VkDeviceSize vbytes = (VkDeviceSize)assets->vert_count * sizeof(DemoVertex);
    VkDeviceSize ibytes = (VkDeviceSize)assets->index_count * sizeof(uint32_t);
    if (!vkc_create_host_buffer(ctx, vbytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &m->vbuf, &m->vmem, &p)) return 0;
    memcpy(p, assets->verts, (size_t)vbytes);
    if (!vkc_create_host_buffer(ctx, ibytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &m->ibuf, &m->imem, &p)) {
        shared_mesh_destroy(m, ctx);   /* unwind the already-created vertex buffer */
        return 0;
    }
    memcpy(p, assets->indices, (size_t)ibytes);
    m->index_count = assets->index_count;
    m->vert_count = assets->vert_count;
    m->cull_back = assets->mesh_cull_back;
    return 1;
}

void shared_mesh_destroy(SharedMesh *m, VkCtx *ctx) {
    vkc_destroy_buffer(ctx, m->vbuf, m->vmem);
    vkc_destroy_buffer(ctx, m->ibuf, m->imem);
    memset(m, 0, sizeof *m);
}

/* ------------------------------------------------------------------ bone-line skeleton geometry */

/* Bind-model-space origin of a joint = translation of inverse(inverse_bind). The canonical skinning
 * palette is M = model_pose . inverse_bind, so M . bind_origin = model_pose . 0 = the posed joint
 * origin (inverse_bind cancels bind_model). Holds for any rig, including an identity inverse-bind
 * (then bind_origin is 0 and M . 0 is still the joint origin). inverse_bind is a 3x4 affine [R|t]
 * row-major; the inverse translation is -R^-1 . t. */
static void bind_origin_from_inverse_bind(const float ib[12], float out[3]) {
    float a = ib[0], b = ib[1], c = ib[2];
    float d = ib[4], e = ib[5], f = ib[6];
    float g = ib[8], h = ib[9], i = ib[10];
    float A = (e*i - f*h), B = -(d*i - f*g), C = (d*h - e*g);
    float det = a*A + b*B + c*C;
    float s = det != 0.0f ? 1.0f / det : 0.0f;
    /* rows of R^-1 = adjugate^T / det */
    float r00 = A*s,            r01 = -(b*i - c*h)*s, r02 =  (b*f - c*e)*s;
    float r10 = B*s,            r11 =  (a*i - c*g)*s, r12 = -(a*f - c*d)*s;
    float r20 = C*s,            r21 = -(a*h - b*g)*s, r22 =  (a*e - b*d)*s;
    float tx = ib[3], ty = ib[7], tz = ib[11];
    out[0] = -(r00*tx + r01*ty + r02*tz);
    out[1] = -(r10*tx + r11*ty + r12*tz);
    out[2] = -(r20*tx + r21*ty + r22*tz);
}

/* Build the static line VB: one LINE_LIST segment (two endpoints) per non-root joint, joint->parent.
 * Geometry is shared by every instance; only the per-instance palette row differs at draw time. */
static int build_skeleton_vb(Crowd *cr, VkCtx *ctx, const mrw_skeleton_view *skel) {
    uint32_t jc = skel->joint_count;
    float *org = (float *)malloc((size_t)jc * 3 * sizeof(float));
    SkelVertex *v = (SkelVertex *)malloc((size_t)jc * 2 * sizeof(SkelVertex));  /* <= 2 per joint */
    if (!org || !v) { free(org); free(v); return 0; }

    for (uint32_t j = 0; j < jc; ++j) {
        float ib[12];
        mrw_skeleton_inverse_bind(skel, j, ib);
        bind_origin_from_inverse_bind(ib, &org[j * 3]);
    }
    uint32_t n = 0;
    for (uint32_t j = 0; j < jc; ++j) {
        uint16_t p = 0xFFFF;
        mrw_skeleton_parent(skel, j, &p);
        if (p == 0xFFFFu) continue;   /* root has no bone segment */
        v[n].joint = j; memcpy(v[n].origin, &org[(size_t)j * 3], 3 * sizeof(float)); n++;
        v[n].joint = p; memcpy(v[n].origin, &org[(size_t)p * 3], 3 * sizeof(float)); n++;
    }
    free(org);
    cr->skel_vert_count = n;
    if (n == 0) { free(v); return 1; }   /* single-joint rig: no bones; draw becomes a no-op */

    void *map;
    VkDeviceSize bytes = (VkDeviceSize)n * sizeof(SkelVertex);
    if (!vkc_create_host_buffer(ctx, bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                &cr->skel_vbuf, &cr->skel_vmem, &map)) { free(v); return 0; }
    memcpy(map, v, (size_t)bytes);
    free(v);
    return 1;
}

static int create_bindless(Crowd *cr, VkCtx *ctx) {
    VkSamplerCreateInfo si = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(ctx->device, &si, NULL, &cr->sampler) != VK_SUCCESS) return 0;

    VkDescriptorSetLayoutBinding b = { 0 };
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = MAX_PALETTES;
    b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorBindingFlags bflags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                      VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bf = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
    bf.bindingCount = 1; bf.pBindingFlags = &bflags;
    VkDescriptorSetLayoutCreateInfo li = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    li.pNext = &bf;
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    li.bindingCount = 1; li.pBindings = &b;
    if (vkCreateDescriptorSetLayout(ctx->device, &li, NULL, &cr->set_layout) != VK_SUCCESS) return 0;

    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_PALETTES };
    VkDescriptorPoolCreateInfo pi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pi.maxSets = 1; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(ctx->device, &pi, NULL, &cr->desc_pool) != VK_SUCCESS) return 0;

    uint32_t var_count = 1;   /* one palette texture in use */
    VkDescriptorSetVariableDescriptorCountAllocateInfo va = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO };
    va.descriptorSetCount = 1; va.pDescriptorCounts = &var_count;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.pNext = &va; ai.descriptorPool = cr->desc_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &cr->set_layout;
    if (vkAllocateDescriptorSets(ctx->device, &ai, &cr->set) != VK_SUCCESS) return 0;

    VkDescriptorImageInfo img = { cr->sampler, cr->palette_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = cr->set; w.dstBinding = 0; w.dstArrayElement = 0;
    w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &img;
    vkUpdateDescriptorSets(ctx->device, 1, &w, 0, NULL);
    return 1;
}

static void init_instances(Crowd *cr) {
    uint32_t cols = (uint32_t)ceilf(sqrtf((float)cr->count));
    float spacing = 1.6f;
    float ox = -0.5f * (float)(cols - 1) * spacing;
    /* push the whole field behind the origin so the foreground belongs to the CPU-animated heroes */
    float oz = -8.0f - (float)(cols - 1) * spacing;
    for (uint32_t i = 0; i < cr->count; ++i) {
        uint32_t cx = i % cols, cz = i / cols;
        InstanceAnim *a = &cr->cpu[i];
        memset(a, 0, sizeof *a);
        mat4 m = mat4_translate(v3(ox + cx * spacing, 0.0f, oz + cz * spacing));
        memcpy(a->model, m.m, sizeof m.m);

        uint32_t clip = (i * 7u + cz) % cr->clip_count;
        a->clipA[0] = cr->clips[clip].first_frame;
        a->clipA[1] = cr->clips[clip].frame_count;
        a->clipA[2] = cr->clips[clip].looping;
        a->clipA[3] = 0;                              /* palette index 0 */
        a->clipB[3] = i;                              /* stable entity id -> tint (crowd.vert) */
        a->times[0] = fmodf((float)i * 0.137f, cr->clips[clip].duration);
        a->times[2] = cr->clips[clip].duration;
        a->blend[0] = 0.0f;                           /* single clip (steady state) */
    }
}

int crowd_init(Crowd *cr, VkCtx *ctx, const ProcAssets *assets, const SharedMesh *mesh,
               uint32_t capacity, uint32_t count) {
    memset(cr, 0, sizeof *cr);
    if (capacity < 1) capacity = 1;
    if (count > capacity) count = capacity;
    cr->capacity = capacity;
    cr->count = count;
    cr->mesh = mesh;

    mrw_blob blob;
    if (mrw_blob_open(assets->blob, assets->blob_size, &blob) != MRW_OK) { fprintf(stderr, "[crowd] blob open\n"); goto fail; }
    mrw_baked_view bv;
    if (!find_baked(&blob, &bv)) { fprintf(stderr, "[crowd] no BAKED section\n"); goto fail; }

    /* clips[] holds at most DEMO_PROC_MAX_CLIPS; clamp so init_instances' `% clip_count` stays in
     * range (a glTF rig may bake more clips than the procedural biped's 3 - all frames still live in
     * the palette texture; instances just cycle through the first few clips). */
    cr->clip_count = bv.clip_count < DEMO_PROC_MAX_CLIPS ? bv.clip_count : (uint32_t)DEMO_PROC_MAX_CLIPS;
    for (uint32_t i = 0; i < cr->clip_count; ++i) {
        mrw_baked_clip entry;
        mrw_baked_clip_entry(&bv, i, &entry);
        cr->clips[i].first_frame = entry.first_frame;
        cr->clips[i].frame_count = entry.frame_count;
        cr->clips[i].looping = (entry.flags & MRW_BAKED_CLIP_LOOPING) ? 1u : 0u;
        cr->clips[i].duration = entry.source_duration_s;
    }

    /* upload baked palette: RGBA16F, width=frame_stride_texels, height=total_frames */
    const void *texels = bv.base + bv.texels_off;
    if (!vkc_create_texture_rgba16f(ctx, bv.frame_stride_texels, bv.total_frames, texels,
                                    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                    &cr->palette_img, &cr->palette_mem, &cr->palette_view)) goto fail;
    fprintf(stderr, "[crowd] palette texture %ux%u RGBA16F (%u texels)\n",
            bv.frame_stride_texels, bv.total_frames, bv.texel_count);

    if (!create_bindless(cr, ctx)) { fprintf(stderr, "[crowd] bindless setup failed\n"); goto fail; }

    /* instance buffers - one per frame-in-flight (host-visible, device address), sized to capacity
     * so the live count can grow without reallocation; only `count` are uploaded/drawn. */
    cr->cpu = (InstanceAnim *)malloc((size_t)capacity * sizeof(InstanceAnim));
    if (!cr->cpu) { fprintf(stderr, "[crowd] instance mirror alloc failed\n"); goto fail; }
    init_instances(cr);
    VkDeviceSize ibuf_bytes = (VkDeviceSize)capacity * sizeof(InstanceAnim);
    for (uint32_t f = 0; f < VKC_FRAMES_IN_FLIGHT; ++f) {
        if (!vkc_create_host_buffer(ctx, ibuf_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                &cr->inst_buf[f], &cr->inst_mem[f], &cr->inst_map[f])) goto fail;
        cr->inst_addr[f] = vkc_buffer_address(ctx, cr->inst_buf[f]);
        memcpy(cr->inst_map[f], cr->cpu, (size_t)count * sizeof(InstanceAnim));
    }

    /* pipeline layout + shader objects */
    VkPushConstantRange pr = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(CrowdPush) };
    VkPipelineLayoutCreateInfo li = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.setLayoutCount = 1; li.pSetLayouts = &cr->set_layout;
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(ctx->device, &li, NULL, &cr->layout) != VK_SUCCESS) goto fail;

    if (!vkc_create_graphics_shaders(ctx, crowd_vert_spv, sizeof crowd_vert_spv,
            crowd_frag_spv, sizeof crowd_frag_spv, &cr->set_layout, 1, &pr, &cr->vs, &cr->fs)) goto fail;
    /* bench-only vertex-only variants (bound with no fragment stage under rasterizer discard) */
    if (!vkc_create_vertex_shader(ctx, crowd_skin_bench_vert_spv, sizeof crowd_skin_bench_vert_spv,
            &cr->set_layout, 1, &pr, &cr->vs_skin_bench)) goto fail;
    if (!vkc_create_vertex_shader(ctx, crowd_static_bench_vert_spv, sizeof crowd_static_bench_vert_spv,
            &cr->set_layout, 1, &pr, &cr->vs_static_bench)) goto fail;

    /* bone-line skeleton render LOD: static line geometry + a VS/FS pair sharing this layout + the
     * bindless palette set (the skeleton poses from the same baked palette the mesh skins from). */
    mrw_skeleton_view skel;
    if (mrw_blob_skeleton(&blob, &skel) != MRW_OK) { fprintf(stderr, "[crowd] no SKELETON section\n"); goto fail; }
    if (!build_skeleton_vb(cr, ctx, &skel)) { fprintf(stderr, "[crowd] skeleton VB build failed\n"); goto fail; }
    if (!vkc_create_graphics_shaders(ctx, crowd_skel_vert_spv, sizeof crowd_skel_vert_spv,
            skel_frag_spv, sizeof skel_frag_spv, &cr->set_layout, 1, &pr, &cr->vs_skel, &cr->fs_skel)) goto fail;

    fprintf(stderr, "[crowd] %u instances (capacity %u)\n", count, capacity);
    return 1;
fail:   /* unwind every partially-created Vulkan resource (crowd_destroy is guarded + idempotent) */
    crowd_destroy(cr, ctx);
    return 0;
}

void crowd_set_count(Crowd *cr, uint32_t count) {
    if (count < 1) count = 1;
    if (count > cr->capacity) count = cr->capacity;
    cr->count = count;
    init_instances(cr);   /* rebuild the grid; crowd_update re-uploads the active range each frame */
}

void crowd_update(Crowd *cr, float dt, uint32_t frame) {
    for (uint32_t i = 0; i < cr->count; ++i) cr->cpu[i].times[0] += dt;   /* advance clip A time */
    memcpy(cr->inst_map[frame], cr->cpu, (size_t)cr->count * sizeof(InstanceAnim));
}

void crowd_draw(Crowd *cr, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent,
                CrowdDrawMode mode) {
    crowd_draw_instances(cr, ctx, cmd, view_proj, extent, cr->inst_addr[ctx->cur_frame], cr->count, mode);
}

void crowd_draw_instances(Crowd *cr, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj,
                          VkExtent2D extent, VkDeviceAddress inst_addr, uint32_t count,
                          CrowdDrawMode mode) {
    VkVertexInputBindingDescription2EXT b = { VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT };
    b.binding = 0; b.stride = sizeof(DemoVertex); b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; b.divisor = 1;
    VkVertexInputAttributeDescription2EXT a[5];
    for (int i = 0; i < 5; ++i) { a[i] = (VkVertexInputAttributeDescription2EXT){ VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT }; a[i].binding = 0; a[i].location = (uint32_t)i; }
    a[0].format = VK_FORMAT_R32G32B32_SFLOAT;    a[0].offset = offsetof(DemoVertex, pos);
    a[1].format = VK_FORMAT_R32G32B32_SFLOAT;    a[1].offset = offsetof(DemoVertex, nrm);
    a[2].format = VK_FORMAT_R32G32B32A32_SFLOAT; a[2].offset = offsetof(DemoVertex, tan);
    a[3].format = VK_FORMAT_R32G32B32A32_UINT;   a[3].offset = offsetof(DemoVertex, bones);
    a[4].format = VK_FORMAT_R32G32B32A32_SFLOAT; a[4].offset = offsetof(DemoVertex, weights);

    /* Discard modes bind a bench VS with NO fragment stage; FULL binds the real linked vs+fs. */
    VkShaderEXT vs = (mode == CROWD_DRAW_SKIN_DISCARD)   ? cr->vs_skin_bench
                   : (mode == CROWD_DRAW_STATIC_DISCARD) ? cr->vs_static_bench : cr->vs;
    VkShaderEXT fs = (mode == CROWD_DRAW_FULL) ? cr->fs : VK_NULL_HANDLE;
    vkc_bind_shaders(ctx, cmd, vs, fs);
    vkc_set_default_state(ctx, cmd, extent, &b, 1, a, 5);
    if (mode == CROWD_DRAW_FULL)
        vkCmdSetCullMode(cmd, cr->mesh->cull_back ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE);
    else
        vkCmdSetRasterizerDiscardEnable(cmd, VK_TRUE);   /* vertex-only timing: no raster/fragment/ROP */
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cr->layout, 0, 1, &cr->set, 0, NULL);

    CrowdPush pc;
    memcpy(pc.viewProj, view_proj->m, sizeof pc.viewProj);
    pc.instances = inst_addr;
    pc.texelsPerBone = 2;
    pc.benchEps = (mode == CROWD_DRAW_FULL) ? 0.0f : 1.0f;   /* nonzero -> skinning can't be DCE'd */
    vkCmdPushConstants(cmd, cr->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof pc, &pc);

    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cr->mesh->vbuf, &zero);
    vkCmdBindIndexBuffer(cmd, cr->mesh->ibuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, cr->mesh->index_count, count, 0, 0, 0);
}

void crowd_draw_skeleton(Crowd *cr, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj,
                         VkExtent2D extent, VkDeviceAddress inst_addr, uint32_t count) {
    if (!cr->skel_vbuf || cr->skel_vert_count == 0) return;   /* rig with no bones */

    VkVertexInputBindingDescription2EXT b = { VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT };
    b.binding = 0; b.stride = sizeof(SkelVertex); b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; b.divisor = 1;
    VkVertexInputAttributeDescription2EXT a[2];
    for (int i = 0; i < 2; ++i) { a[i] = (VkVertexInputAttributeDescription2EXT){ VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT }; a[i].binding = 0; a[i].location = (uint32_t)i; }
    a[0].format = VK_FORMAT_R32_UINT;          a[0].offset = offsetof(SkelVertex, joint);
    a[1].format = VK_FORMAT_R32G32B32_SFLOAT;  a[1].offset = offsetof(SkelVertex, origin);

    vkc_bind_shaders(ctx, cmd, cr->vs_skel, cr->fs_skel);
    vkc_set_default_state(ctx, cmd, extent, &b, 1, a, 2);
    vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);   /* the one state lines flip; width is already 1.0 */
    vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);                          /* face culling is meaningless for lines */
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cr->layout, 0, 1, &cr->set, 0, NULL);

    CrowdPush pc;
    memcpy(pc.viewProj, view_proj->m, sizeof pc.viewProj);
    pc.instances = inst_addr;
    pc.texelsPerBone = 2;
    pc.benchEps = 0.0f;   /* unused by crowd_skel.vert (no bench variant); kept for layout parity */
    vkCmdPushConstants(cmd, cr->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof pc, &pc);

    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cr->skel_vbuf, &zero);
    vkCmdDraw(cmd, cr->skel_vert_count, count, 0, 0);
}

void crowd_destroy(Crowd *cr, VkCtx *ctx) {
    vkDeviceWaitIdle(ctx->device);
    vkc_destroy_shader(ctx, cr->vs); vkc_destroy_shader(ctx, cr->fs);
    vkc_destroy_shader(ctx, cr->vs_skin_bench); vkc_destroy_shader(ctx, cr->vs_static_bench);
    vkc_destroy_shader(ctx, cr->vs_skel); vkc_destroy_shader(ctx, cr->fs_skel);
    vkc_destroy_buffer(ctx, cr->skel_vbuf, cr->skel_vmem);
    if (cr->layout) vkDestroyPipelineLayout(ctx->device, cr->layout, NULL);
    if (cr->desc_pool) vkDestroyDescriptorPool(ctx->device, cr->desc_pool, NULL);
    if (cr->set_layout) vkDestroyDescriptorSetLayout(ctx->device, cr->set_layout, NULL);
    if (cr->sampler) vkDestroySampler(ctx->device, cr->sampler, NULL);
    if (cr->palette_view) vkDestroyImageView(ctx->device, cr->palette_view, NULL);
    if (cr->palette_img) vkDestroyImage(ctx->device, cr->palette_img, NULL);
    if (cr->palette_mem) vkFreeMemory(ctx->device, cr->palette_mem, NULL);
    /* the skinned mesh is owned by main (SharedMesh) - not freed here */
    for (uint32_t f = 0; f < VKC_FRAMES_IN_FLIGHT; ++f)
        vkc_destroy_buffer(ctx, cr->inst_buf[f], cr->inst_mem[f]);
    free(cr->cpu);
    memset(cr, 0, sizeof *cr);
}
