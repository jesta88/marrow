#include "field.h"

#include "mrw_authoring.h"   /* mrw_authoring_alloc/free - 64-aligned cacheable scratch for the kernel */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

/* Mirrors skin_tierA.vert's HeroInstance (scalar layout): mat4 model + vec4 tint. */
typedef struct { float model[16]; float tint[4]; } FieldInstance;
_Static_assert(sizeof(FieldInstance) == 80, "FieldInstance must be 80 bytes");

/* Mirrors skin_tierA.vert's push constant (== HeroPush). */
typedef struct {
    float           viewProj[16];
    VkDeviceAddress heroes;
    VkDeviceAddress palettes;
    uint32_t        jointCount;
    uint32_t        _pad;
} FieldPush;
_Static_assert(sizeof(FieldPush) == 88, "FieldPush must be 88 bytes");

/* near candidate for the deterministic top-K (nearest near_cap by distance, id-tie-broken). */
typedef struct { float d2; uint32_t id; uint32_t idx; } Cand;

/* Mirrors crowd.c's SkelVertex: the layout of the borrowed crowd's static bone-line VB, which the
 * near skeleton draw reuses (one endpoint = {joint, bind-model-space origin}). */
typedef struct { uint32_t joint; float origin[3]; } FieldSkelVertex;
_Static_assert(sizeof(FieldSkelVertex) == 16, "FieldSkelVertex must be 16 bytes");

#include "skin_tierA_vert.spv.h"
#include "skin_tierA_skel_vert.spv.h"   /* near bone-line skeleton (global all-skeleton toggle) */
#include "crowd_frag.spv.h"   /* reuse the crowd fragment (same vNormal/vColor inputs) */
#include "skel_frag.spv.h"    /* flat per-instance tint for the bone lines */

/* crowd.vert's per-instance tint, computed CPU-side from the stable id so the near Tier-A color
 * matches the far Tier-B color for the same entity (the seamless-promotion claim). */
static void field_tint(uint32_t id, float out[4]) {
    out[0] = 0.45f + 0.45f * cosf((float)id * 0.91f);
    out[1] = 0.45f + 0.45f * cosf((float)id * 1.73f + 2.0f);
    out[2] = 0.45f + 0.45f * cosf((float)id * 2.21f + 4.0f);
    out[3] = 1.0f;
}

static const char *backend_name(mrw_backend b) {
    return b == MRW_BACKEND_AVX2 ? "AVX2" : (b == MRW_BACKEND_SSE2 ? "SSE2" : "scalar");
}

static int find_baked(const mrw_blob *b, mrw_baked_view *out) {
    for (uint32_t i = 0; i < b->section_count; ++i) {
        uint32_t type = 0;
        if (mrw_blob_section_type(b, i, &type) == MRW_OK && type == MRW_SECTION_BAKED)
            return mrw_baked_view_at(b, i, out) == MRW_OK;
    }
    return 0;
}

/* (Re)lay out the live grid: a square field receding away from the camera in -Z. Each entity gets a
 * fixed position, a stable id, a cross-fade clip PAIR (cycled for crowd variety; a single locomotion
 * pair on a 2-clip rig), and a desynchronized initial phase. The persistent LOD state is reset to
 * Tier-B/mesh so the first partition promotes the near band cleanly. */
static void field_layout(Field *f) {
    uint32_t count = f->count;
    uint32_t cols = (uint32_t)ceilf(sqrtf((float)count));
    if (cols < 1) cols = 1;
    const float spacing = 1.6f;
    const float front_z = 18.0f;                 /* nearest row, just behind the camera's foreground */
    float ox = -0.5f * (float)(cols - 1) * spacing;

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t cx = i % cols, cz = i / cols;
        FieldEntity *e = &f->ent[i];
        e->pos[0] = ox + (float)cx * spacing;
        e->pos[1] = 0.0f;
        e->pos[2] = front_z - (float)cz * spacing;
        e->id = i;
        if (f->clip_count <= 1) {
            e->clipA = e->clipB = 0;
        } else {
            uint32_t k = (i * 7u + cz) % f->clip_count;
            e->clipA = (uint16_t)k;
            e->clipB = (uint16_t)((k + 1u) % f->clip_count);
        }
        float dur = f->clip_dur[e->clipA] > 0.0f ? f->clip_dur[e->clipA] : 1.0f;
        e->phase = fmodf((float)i * 0.137f, dur);
        e->w = 0.0f;
        f->anim_tier[i] = 1;    /* start Tier B; the partition promotes the near band */
        f->render_lod[i] = 0;   /* start mesh                                          */
    }
}

int field_init(Field *f, VkCtx *ctx, const ProcAssets *assets, Crowd *crowd,
               uint32_t total_capacity, uint32_t near_cap, uint32_t count, FieldLod lod) {
    memset(f, 0, sizeof *f);
    if (total_capacity < 1) total_capacity = 1;
    if (near_cap < 1) near_cap = 1;
    if (near_cap > total_capacity) near_cap = total_capacity;
    if (count > total_capacity) count = total_capacity;
    if (count < 1) count = 1;
    f->crowd = crowd;
    f->total_capacity = total_capacity;
    f->near_cap = near_cap;
    f->count = count;
    f->joint_count = assets->joint_count;
    f->lod = lod;

    if (mrw_blob_open(assets->blob, assets->blob_size, &f->blob) != MRW_OK) {
        fprintf(stderr, "[field] blob open\n"); goto fail;
    }
    mrw_blob_skeleton(&f->blob, &f->skel);
    mrw_baked_view bv;
    if (!find_baked(&f->blob, &bv)) { fprintf(stderr, "[field] no BAKED section (field needs Tier B)\n"); goto fail; }

    /* Resolve the clip set: a dense CLIP view (Tier A) + the baked clip-table entry (Tier B) per clip.
     * The two tiers cross-fade the SAME pair, so an entity animates identically in either - the
     * promotion is seamless. */
    f->clip_count = assets->clip_count < DEMO_PROC_MAX_CLIPS ? assets->clip_count
                                                             : (uint32_t)DEMO_PROC_MAX_CLIPS;
    if (f->clip_count == 0) { fprintf(stderr, "[field] no clips\n"); goto fail; }
    for (uint32_t k = 0; k < f->clip_count; ++k) {
        uint32_t ci = assets->clips[k].clip_index;
        if (mrw_clip_view_at(&f->blob, 1 + ci, &f->clipv[k]) != MRW_OK) {
            fprintf(stderr, "[field] clip %u view failed\n", k); goto fail;
        }
        mrw_baked_clip entry;
        mrw_baked_clip_entry(&bv, ci, &entry);
        f->baked_first[k] = entry.first_frame;
        f->baked_count[k] = entry.frame_count;
        f->baked_loop[k]  = (entry.flags & MRW_BAKED_CLIP_LOOPING) ? 1u : 0u;
        f->clip_dur[k]    = assets->clips[k].duration_s;
    }

    uint32_t jc = f->joint_count;

    /* canonical field + persistent LOD state + partition scratch (all total_capacity-sized) */
    f->ent        = (FieldEntity *)malloc((size_t)total_capacity * sizeof(FieldEntity));
    f->anim_tier  = (uint8_t *)malloc((size_t)total_capacity);
    f->render_lod = (uint8_t *)malloc((size_t)total_capacity);
    f->near_mark  = (uint8_t *)malloc((size_t)total_capacity);
    f->cand       = malloc((size_t)total_capacity * sizeof(Cand));
    f->far_stage  = (InstanceAnim *)malloc((size_t)total_capacity * sizeof(InstanceAnim));
    if (!f->ent || !f->anim_tier || !f->render_lod || !f->near_mark || !f->cand || !f->far_stage) {
        fprintf(stderr, "[field] canonical alloc failed\n"); goto fail;
    }

    /* near Tier-A CPU scratch (near_cap-sized contiguous batch inputs + the cacheable kernel output) */
    f->timesA  = (float *)malloc((size_t)near_cap * sizeof(float));
    f->timesB  = (float *)malloc((size_t)near_cap * sizeof(float));
    f->weights = (float *)malloc((size_t)near_cap * sizeof(float));
    f->hero_stage = malloc((size_t)near_cap * sizeof(FieldInstance));
    if (!f->timesA || !f->timesB || !f->weights || !f->hero_stage) {
        fprintf(stderr, "[field] near scratch alloc failed\n"); goto fail;
    }
    mrw_mem_req sreq, preq;
    if (mrw_batch_blend_clips_to_palette_requirements(jc, near_cap, MRW_PALETTE_F32, &sreq, &preq) != MRW_OK) {
        fprintf(stderr, "[field] batch requirements failed\n"); goto fail;
    }
    f->pal_bytes = preq.size; f->batch_bytes = sreq.size;
    f->pal_scratch   = (float *)mrw_authoring_alloc(f->pal_bytes);
    f->batch_scratch = mrw_authoring_alloc(f->batch_bytes);
    if (!f->pal_scratch || !f->batch_scratch) { fprintf(stderr, "[field] kernel scratch alloc failed\n"); goto fail; }

    /* near GPU buffers (model+tint upload + f32 palette SSBO) + far GPU buffers (InstanceAnim) - one
     * copy per frame-in-flight so the CPU never overwrites a buffer the GPU is still reading. */
    VkBufferUsageFlags su = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkDeviceSize hero_bytes = (VkDeviceSize)near_cap * sizeof(FieldInstance);
    VkDeviceSize far_bytes  = (VkDeviceSize)total_capacity * sizeof(InstanceAnim);
    for (uint32_t fr = 0; fr < VKC_FRAMES_IN_FLIGHT; ++fr) {
        if (!vkc_create_host_buffer(ctx, hero_bytes, su, &f->hero_buf[fr], &f->hero_mem[fr], &f->hero_map[fr])) goto fail;
        f->hero_addr[fr] = vkc_buffer_address(ctx, f->hero_buf[fr]);
        if (!vkc_create_host_buffer(ctx, (VkDeviceSize)f->pal_bytes, su, &f->pal_buf[fr], &f->pal_mem[fr], &f->pal_map[fr])) goto fail;
        f->pal_addr[fr] = vkc_buffer_address(ctx, f->pal_buf[fr]);
        if (!vkc_create_host_buffer(ctx, far_bytes, su, &f->far_buf[fr], &f->far_mem[fr], &f->far_map[fr])) goto fail;
        f->far_addr[fr] = vkc_buffer_address(ctx, f->far_buf[fr]);
    }

    /* near pipeline: the Tier-A skinning VS + the shared crowd fragment, on its own layout (the same
     * scalar push the heroes use). The far tier reuses the borrowed Crowd's pipeline. */
    VkPushConstantRange pr = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(FieldPush) };
    VkPipelineLayoutCreateInfo li = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(ctx->device, &li, NULL, &f->layout) != VK_SUCCESS) goto fail;
    if (!vkc_create_graphics_shaders(ctx, skin_tierA_vert_spv, sizeof skin_tierA_vert_spv,
            crowd_frag_spv, sizeof crowd_frag_spv, NULL, 0, &pr, &f->vs, &f->fs)) goto fail;
    /* bone-line variant of the near pipeline: the Tier-A skeleton VS poses joint origins from the same
     * CPU palette SSBO the mesh skins from, sharing this layout; a flat fragment (no normals). */
    if (!vkc_create_graphics_shaders(ctx, skin_tierA_skel_vert_spv, sizeof skin_tierA_skel_vert_spv,
            skel_frag_spv, sizeof skel_frag_spv, NULL, 0, &pr, &f->vs_skel, &f->fs_skel)) goto fail;

    mrw_dispatch_detect(&f->disp);
    f->backend_name = backend_name(f->disp.backend);

    field_layout(f);

    fprintf(stderr, "[field] %u entities (capacity %u, near_cap %u), %u clip(s), near backend %s\n",
            count, total_capacity, near_cap, f->clip_count, f->backend_name);
    return 1;
fail:
    field_destroy(f, ctx);
    return 0;
}

void field_set_count(Field *f, uint32_t count) {
    if (count < 1) count = 1;
    if (count > f->total_capacity) count = f->total_capacity;
    f->count = count;
    field_layout(f);
}

static int cand_cmp(const void *pa, const void *pb) {
    const Cand *a = (const Cand *)pa, *b = (const Cand *)pb;
    if (a->d2 < b->d2) return -1;
    if (a->d2 > b->d2) return  1;
    return a->id < b->id ? -1 : (a->id > b->id ? 1 : 0);   /* stable, deterministic tie-break */
}

void field_update(Field *f, const float cam_pos[3], float dt, uint32_t frame, Profiler *prof) {
    uint32_t jc = f->joint_count, count = f->count, cc = f->clip_count;
    Cand *cand = (Cand *)f->cand;
    FieldInstance *hero = (FieldInstance *)f->hero_stage;

    /* (1) advance the animation: monotonic phase + an oscillating walk<->run weight, desynced by id */
    prof_begin(prof, PROF_CROWD_UPDATE);
    for (uint32_t i = 0; i < count; ++i) {
        FieldEntity *e = &f->ent[i];
        e->phase += dt;
        e->w = 0.5f + 0.5f * sinf(e->phase * 0.6f + (float)e->id * 0.1f);
    }
    prof_end(prof, PROF_CROWD_UPDATE);

    /* (2) PROF_LOD: partition by distance (hysteresis), select the nearest near_cap as the Tier-A set
     * (deterministic top-K), then compact near (counting-sorted by clip pair) + far (the rest). */
    prof_begin(prof, PROF_LOD);

    const float a_enter = f->lod.r_a, a_exit = f->lod.r_a * 1.10f;
    const float m_enter = f->lod.r_mesh, m_exit = f->lod.r_mesh * 1.10f;
    const float a_enter2 = a_enter * a_enter, a_exit2 = a_exit * a_exit;
    const float m_enter2 = m_enter * m_enter, m_exit2 = m_exit * m_exit;

    uint32_t ncand = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const FieldEntity *e = &f->ent[i];
        float dx = e->pos[0] - cam_pos[0], dy = e->pos[1] - cam_pos[1], dz = e->pos[2] - cam_pos[2];
        float d2 = dx*dx + dy*dy + dz*dz;
        /* hysteresis: cross the inner radius to promote, the (wider) outer radius to demote */
        if (f->anim_tier[i]) { if (d2 < a_enter2) f->anim_tier[i] = 0; }
        else                 { if (d2 > a_exit2)  f->anim_tier[i] = 1; }
        if (f->render_lod[i]) { if (d2 < m_enter2) f->render_lod[i] = 0; }
        else                  { if (d2 > m_exit2)  f->render_lod[i] = 1; }
        if (f->anim_tier[i] == 0) {
            cand[ncand].d2 = d2; cand[ncand].id = e->id; cand[ncand].idx = i; ncand++;
        }
    }

    /* top-K: keep the nearest near_cap Tier-A candidates; the rest fall back to Tier B (bounded and
     * honest). Only sort on overflow - in the tuned common case the whole candidate set fits. */
    uint32_t nsel = ncand;
    f->near_clamped = 0;
    if (ncand > f->near_cap) {
        qsort(cand, ncand, sizeof(Cand), cand_cmp);
        nsel = f->near_cap;
        f->near_clamped = ncand - f->near_cap;
    }
    memset(f->near_mark, 0, count);
    for (uint32_t s = 0; s < nsel; ++s) f->near_mark[cand[s].idx] = 1;

    /* counting-sort the near set by (clipA,clipB) into contiguous homogeneous groups, so the batched
     * blend runs once per pair (keys are clipA*cc+clipB, <= cc*cc bins); in the same pass count the far
     * full-mesh entities so the far set can be compacted into a mesh sub-range then a skeleton tail. */
    uint32_t kcount[FIELD_MAX_GROUPS] = { 0 };
    uint32_t far_mesh = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (f->near_mark[i]) kcount[f->ent[i].clipA * cc + f->ent[i].clipB]++;
        else if (!f->skeleton_all && f->render_lod[i] == 0) far_mesh++;
    }
    uint32_t cursor[FIELD_MAX_GROUPS], acc = 0;
    f->group_count = 0;
    for (uint32_t key = 0; key < cc * cc; ++key) {
        cursor[key] = acc;
        if (kcount[key]) {
            f->groups[f->group_count].clipA = (uint16_t)(key / cc);
            f->groups[f->group_count].clipB = (uint16_t)(key % cc);
            f->groups[f->group_count].start = acc;
            f->groups[f->group_count].count = kcount[key];
            f->group_count++;
        }
        acc += kcount[key];
    }
    f->near_count = acc;

    /* compact: near entities scatter into their sorted group slot (model+tint+batch inputs); far
     * entities append into the InstanceAnim stage carrying the stable id (clipB.w) for the tint. The
     * far set splits by render LOD into a full-mesh band [0, far_mesh) then a skeleton tail
     * [far_mesh, far_count); the global all-skeleton toggle forces every far entity into the tail. */
    uint32_t fm = 0, fs = far_mesh;
    for (uint32_t i = 0; i < count; ++i) {
        const FieldEntity *e = &f->ent[i];
        mat4 model = mat4_translate(v3(e->pos[0], e->pos[1], e->pos[2]));
        if (f->near_mark[i]) {
            uint32_t dst = cursor[e->clipA * cc + e->clipB]++;
            memcpy(hero[dst].model, model.m, sizeof model.m);
            field_tint(e->id, hero[dst].tint);
            f->timesA[dst]  = e->phase;
            f->timesB[dst]  = e->phase;
            f->weights[dst] = e->w;
        } else {
            uint32_t dst = (!f->skeleton_all && f->render_lod[i] == 0) ? fm++ : fs++;
            InstanceAnim *a = &f->far_stage[dst];
            memcpy(a->model, model.m, sizeof model.m);
            uint32_t cA = e->clipA, cB = e->clipB;
            a->clipA[0] = f->baked_first[cA]; a->clipA[1] = f->baked_count[cA]; a->clipA[2] = f->baked_loop[cA]; a->clipA[3] = 0;
            a->clipB[0] = f->baked_first[cB]; a->clipB[1] = f->baked_count[cB]; a->clipB[2] = f->baked_loop[cB]; a->clipB[3] = e->id;
            a->times[0] = e->phase; a->times[1] = e->phase; a->times[2] = f->clip_dur[cA]; a->times[3] = f->clip_dur[cB];
            a->blend[0] = e->w; a->blend[1] = a->blend[2] = a->blend[3] = 0.0f;
        }
    }
    f->far_mesh_count = far_mesh;
    f->far_skel_count = fs - far_mesh;
    f->far_count = fs;
    prof_end(prof, PROF_LOD);

    /* (3) PALETTE_GEN: one homogeneous batched blend per clip-pair group into the cacheable scratch. */
    prof_begin(prof, PROF_PALETTE_GEN);
    for (uint32_t g = 0; g < f->group_count; ++g) {
        const FieldGroup *grp = &f->groups[g];
        size_t off = (size_t)grp->start * jc * 12u;   /* element offset into the AoS palette */
        mrw_result r = mrw_batch_blend_clips_to_palette(
            &f->disp, &f->skel, &f->clipv[grp->clipA], &f->clipv[grp->clipB],
            &f->timesA[grp->start], &f->timesB[grp->start], &f->weights[grp->start], NULL, grp->count,
            f->pal_scratch + off, f->pal_bytes - off * sizeof(float),
            f->batch_scratch, f->batch_bytes);
        if (r != MRW_OK) { fprintf(stderr, "[field] batch blend group %u failed: %u\n", g, r); break; }
    }
    prof_end(prof, PROF_PALETTE_GEN);

    /* (4) PALETTE_UPLOAD: bulk-copy the cacheable stages into this frame's write-combined GPU buffers. */
    prof_begin(prof, PROF_PALETTE_UPLOAD);
    memcpy(f->pal_map[frame],  f->pal_scratch, (size_t)f->near_count * jc * 12u * sizeof(float));
    memcpy(f->hero_map[frame], hero,           (size_t)f->near_count * sizeof(FieldInstance));
    memcpy(f->far_map[frame],  f->far_stage,   (size_t)f->far_count * sizeof(InstanceAnim));
    prof_end(prof, PROF_PALETTE_UPLOAD);
}

void field_draw_far(Field *f, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent) {
    if (f->far_mesh_count == 0) return;
    crowd_draw_instances(f->crowd, ctx, cmd, view_proj, extent,
                         f->far_addr[ctx->cur_frame], f->far_mesh_count, CROWD_DRAW_FULL);
}

void field_draw_far_skeleton(Field *f, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent) {
    if (f->far_skel_count == 0) return;
    /* the skeleton tail occupies [far_mesh_count, far_count) of the same buffer - offset the base
     * address past the mesh band so the draw's gl_InstanceIndex addresses the tail from 0. */
    VkDeviceAddress base = f->far_addr[ctx->cur_frame] + (VkDeviceSize)f->far_mesh_count * sizeof(InstanceAnim);
    crowd_draw_skeleton(f->crowd, ctx, cmd, view_proj, extent, base, f->far_skel_count);
}

void field_draw_near(Field *f, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent) {
    if (f->near_count == 0 || f->skeleton_all) return;   /* under the global toggle the near set is lines */
    const SharedMesh *mesh = f->crowd->mesh;

    VkVertexInputBindingDescription2EXT b = { VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT };
    b.binding = 0; b.stride = sizeof(DemoVertex); b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; b.divisor = 1;
    VkVertexInputAttributeDescription2EXT a[5];
    for (int i = 0; i < 5; ++i) { a[i] = (VkVertexInputAttributeDescription2EXT){ VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT }; a[i].binding = 0; a[i].location = (uint32_t)i; }
    a[0].format = VK_FORMAT_R32G32B32_SFLOAT;    a[0].offset = offsetof(DemoVertex, pos);
    a[1].format = VK_FORMAT_R32G32B32_SFLOAT;    a[1].offset = offsetof(DemoVertex, nrm);
    a[2].format = VK_FORMAT_R32G32B32A32_SFLOAT; a[2].offset = offsetof(DemoVertex, tan);
    a[3].format = VK_FORMAT_R32G32B32A32_UINT;   a[3].offset = offsetof(DemoVertex, bones);
    a[4].format = VK_FORMAT_R32G32B32A32_SFLOAT; a[4].offset = offsetof(DemoVertex, weights);

    vkc_bind_shaders(ctx, cmd, f->vs, f->fs);
    vkc_set_default_state(ctx, cmd, extent, &b, 1, a, 5);
    vkCmdSetCullMode(cmd, mesh->cull_back ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE);

    FieldPush pc;
    memcpy(pc.viewProj, view_proj->m, sizeof pc.viewProj);
    pc.heroes = f->hero_addr[ctx->cur_frame];
    pc.palettes = f->pal_addr[ctx->cur_frame];
    pc.jointCount = f->joint_count;
    pc._pad = 0;
    vkCmdPushConstants(cmd, f->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof pc, &pc);

    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->vbuf, &zero);
    vkCmdBindIndexBuffer(cmd, mesh->ibuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, mesh->index_count, f->near_count, 0, 0, 0);
}

void field_draw_near_skeleton(Field *f, VkCtx *ctx, VkCommandBuffer cmd, const mat4 *view_proj, VkExtent2D extent) {
    if (!f->skeleton_all || f->near_count == 0) return;
    const Crowd *cr = f->crowd;
    if (!cr->skel_vbuf || cr->skel_vert_count == 0) return;   /* rig with no bones */

    /* same static bone-line VB the far tail uses (borrowed from the crowd), posed instead from the near
     * set's CPU Tier-A palette: skin_tierA_skel.vert fetches M_joint by index and outputs M . restOrigin. */
    VkVertexInputBindingDescription2EXT b = { VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT };
    b.binding = 0; b.stride = sizeof(FieldSkelVertex); b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; b.divisor = 1;
    VkVertexInputAttributeDescription2EXT a[2];
    for (int i = 0; i < 2; ++i) { a[i] = (VkVertexInputAttributeDescription2EXT){ VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT }; a[i].binding = 0; a[i].location = (uint32_t)i; }
    a[0].format = VK_FORMAT_R32_UINT;          a[0].offset = offsetof(FieldSkelVertex, joint);
    a[1].format = VK_FORMAT_R32G32B32_SFLOAT;  a[1].offset = offsetof(FieldSkelVertex, origin);

    vkc_bind_shaders(ctx, cmd, f->vs_skel, f->fs_skel);
    vkc_set_default_state(ctx, cmd, extent, &b, 1, a, 2);
    vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);   /* the one state the lines flip */
    vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);                          /* face culling is meaningless for lines */

    FieldPush pc;
    memcpy(pc.viewProj, view_proj->m, sizeof pc.viewProj);
    pc.heroes = f->hero_addr[ctx->cur_frame];
    pc.palettes = f->pal_addr[ctx->cur_frame];
    pc.jointCount = f->joint_count;
    pc._pad = 0;
    vkCmdPushConstants(cmd, f->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof pc, &pc);

    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cr->skel_vbuf, &zero);
    vkCmdDraw(cmd, cr->skel_vert_count, f->near_count, 0, 0);
}

void field_destroy(Field *f, VkCtx *ctx) {
    vkDeviceWaitIdle(ctx->device);
    vkc_destroy_shader(ctx, f->vs); vkc_destroy_shader(ctx, f->fs);
    vkc_destroy_shader(ctx, f->vs_skel); vkc_destroy_shader(ctx, f->fs_skel);
    if (f->layout) vkDestroyPipelineLayout(ctx->device, f->layout, NULL);
    for (uint32_t fr = 0; fr < VKC_FRAMES_IN_FLIGHT; ++fr) {
        vkc_destroy_buffer(ctx, f->hero_buf[fr], f->hero_mem[fr]);
        vkc_destroy_buffer(ctx, f->pal_buf[fr],  f->pal_mem[fr]);
        vkc_destroy_buffer(ctx, f->far_buf[fr],  f->far_mem[fr]);
    }
    mrw_authoring_free(f->pal_scratch);
    mrw_authoring_free(f->batch_scratch);
    free(f->ent); free(f->anim_tier); free(f->render_lod); free(f->near_mark);
    free(f->cand); free(f->far_stage);
    free(f->timesA); free(f->timesB); free(f->weights); free(f->hero_stage);
    /* the borrowed Crowd + shared mesh are owned by the caller - not freed here */
    memset(f, 0, sizeof *f);
}
