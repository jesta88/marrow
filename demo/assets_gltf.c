/* See assets_gltf.h. Two halves: (1) load + validate the tool-produced .mrw and reconstruct the
 * demo's clip table from its CLIP/BAKED sections; (2) read the source glTF's skinned mesh with
 * cgltf, remapping skin-order JOINTS_0 to marrow joint indices via the lifted DFS so the vertices
 * address the same skeleton the .mrw carries. The marrow runtime is never touched - this is demo
 * code that only consumes the public ABI plus the offline tools' shared helpers. */
#include "assets_gltf.h"

#include "marrow.h"
#include "mrw_authoring.h"   /* mrw_authoring_alloc/free - 64-aligned blob (mrw_blob_open req) */
#include "joint_order.h"     /* mrw_g2m_joint_order - the shared skin-order → marrow-index DFS */
#include "cgltf.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read a whole file into a fresh 64-aligned buffer (mrw_blob_open requires >=64-byte alignment).
 * Freed with mrw_authoring_free, like every other blob in the offline path. */
static int read_file_aligned(const char *path, uint8_t **out_buf, size_t *out_size) {
    *out_buf = NULL; *out_size = 0;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[assets_gltf] cannot open '%s'\n", path); return -1; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long n = ftell(f);
    if (n <= 0) { fclose(f); fprintf(stderr, "[assets_gltf] '%s' is empty or unreadable\n", path); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    uint8_t *buf = (uint8_t *)mrw_authoring_alloc((size_t)n);
    if (!buf) { fclose(f); fprintf(stderr, "[assets_gltf] out of memory reading '%s'\n", path); return -1; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { mrw_authoring_free(buf); fprintf(stderr, "[assets_gltf] short read on '%s'\n", path); return -1; }
    *out_buf = buf; *out_size = (size_t)n;
    return 0;
}

/* Reconstruct the demo clip table from the loaded blob. The crowd reads its clip list straight from
 * the BAKED section, but heroes/validate look up clips by the role names "walk"/"run" in clips[];
 * map those roles onto the available real clips so a single-clip rig still resolves both - walk and
 * run alias clip 0, and the engine cross-fade blends it with itself (a clean no-op). */
static int build_clip_table(const mrw_blob *blob, ProcAssets *out) {
    uint32_t nreal = 0; int has_baked = 0;
    for (uint32_t i = 0; i < blob->section_count; ++i) {
        uint32_t ty = 0;
        if (mrw_blob_section_type(blob, i, &ty) != MRW_OK) continue;
        if (ty == MRW_SECTION_CLIP) ++nreal;
        else if (ty == MRW_SECTION_BAKED) has_baked = 1;
    }
    out->tier_b_eligible = has_baked;
    if (nreal == 0) { fprintf(stderr, "[assets_gltf] .mrw has no CLIP sections\n"); return -1; }

    /* Read metadata for the first min(nreal, DEMO_PROC_MAX_CLIPS) clips (the most we expose by role).
     * Clip ordinal c lives at section 1+c (SKELETON is 0, BAKED last) - the authoring contract the
     * crowd/heroes loaders already rely on (they index sections as 1 + clip_index). */
    uint32_t want = nreal < DEMO_PROC_MAX_CLIPS ? nreal : (uint32_t)DEMO_PROC_MAX_CLIPS;
    struct { float dur; int loop, rm; } meta[DEMO_PROC_MAX_CLIPS];
    uint32_t found = 0;
    for (uint32_t i = 0; i < blob->section_count && found < want; ++i) {
        uint32_t ty = 0;
        if (mrw_blob_section_type(blob, i, &ty) != MRW_OK || ty != MRW_SECTION_CLIP) continue;
        mrw_clip_view cv;
        if (mrw_clip_view_at(blob, i, &cv) != MRW_OK) { fprintf(stderr, "[assets_gltf] unreadable CLIP section\n"); return -1; }
        meta[found].dur  = (cv.sample_count <= 1) ? 0.0f : (float)(cv.sample_count - 1) / cv.fps;
        meta[found].loop = (cv.flags & MRW_CLIP_LOOPING) ? 1 : 0;
        meta[found].rm   = (cv.flags & MRW_CLIP_HAS_ROOT_MOTION) ? 1 : 0;
        ++found;
    }

    static const char *roles[DEMO_PROC_MAX_CLIPS] = { "walk", "run", "clip2", "clip3" };
    if (nreal >= 2) {
        for (uint32_t k = 0; k < want; ++k) {
            out->clips[k].name            = roles[k];
            out->clips[k].clip_index      = k;          /* role k -> real clip k */
            out->clips[k].duration_s      = meta[k].dur;
            out->clips[k].looping         = meta[k].loop;
            out->clips[k].has_root_motion = meta[k].rm;
        }
        out->clip_count = want;
    } else {
        for (uint32_t k = 0; k < 2; ++k) {              /* single real clip aliased to walk + run */
            out->clips[k].name            = roles[k];
            out->clips[k].clip_index      = 0;
            out->clips[k].duration_s      = meta[0].dur;
            out->clips[k].looping         = meta[0].loop;
            out->clips[k].has_root_motion = meta[0].rm;
        }
        out->clip_count = 2;
    }
    return 0;
}

/* Checked cross-check: gltf2marrow stores each glTF joint node's name verbatim as the skeleton joint
 * name, so for the correct .mrw the DFS-remapped names MUST agree. A mismatch (with both names
 * present) means this .mrw was not produced from this glTF - the remap would be silently wrong, so
 * reject. Joints without names (or synthesized "joint_N") are skipped - the DFS remap is authoritative. */
static int cross_check_names(const mrw_skeleton_view *sv, const cgltf_skin *skin,
                             const mrw_g2m_joint_order *jo) {
    for (uint32_t js = 0; js < jo->joint_count; ++js) {
        const char *gn = skin->joints[js] ? skin->joints[js]->name : NULL;
        if (!gn || !gn[0]) continue;
        const char *mn = NULL;
        if (mrw_skeleton_joint_name(sv, jo->marrow_of[js], &mn) != MRW_OK || !mn || !mn[0]) continue;
        if (strcmp(gn, mn) != 0) {
            fprintf(stderr, "[assets_gltf] joint name mismatch at skin joint %u: glTF '%s' vs .mrw '%s' "
                            "(wrong .mrw for this glTF?)\n", js, gn, mn);
            return -1;
        }
    }
    return 0;
}

static cgltf_accessor *find_attr(const cgltf_primitive *p, cgltf_attribute_type ty, int index) {
    for (cgltf_size i = 0; i < p->attributes_count; ++i)
        if (p->attributes[i].type == ty && p->attributes[i].index == index) return p->attributes[i].data;
    return NULL;
}

static int all_finite(const float *v, int n) {
    for (int i = 0; i < n; ++i) if (!isfinite(v[i])) return 0;
    return 1;
}

/* Read every skinned triangle primitive bound to `skin` into out->verts/indices (DemoVertex layout),
 * remapping skin-order JOINTS_0 to marrow indices and renormalizing weights to sum 1. glTF skinning
 * ignores the mesh node's own transform, so POSITION is already in bind space (what the palette
 * expects). NORMAL/TANGENT are optional (defaulted); only pos/normal/bones/weights drive the shaders. */
static int build_mesh(const cgltf_data *data, const cgltf_skin *skin,
                      const mrw_g2m_joint_order *jo, ProcAssets *out) {
    uint32_t jc_skin = jo->joint_count;

    /* pass 1: total vertex + index counts (so we allocate once, no realloc) */
    uint64_t total_v = 0, total_i = 0;
    for (cgltf_size ni = 0; ni < data->nodes_count; ++ni) {
        const cgltf_node *node = &data->nodes[ni];
        if (node->skin != skin || !node->mesh) continue;
        for (cgltf_size pi = 0; pi < node->mesh->primitives_count; ++pi) {
            const cgltf_primitive *prim = &node->mesh->primitives[pi];
            if (prim->type != cgltf_primitive_type_triangles) continue;
            cgltf_accessor *pos = find_attr(prim, cgltf_attribute_type_position, 0);
            cgltf_accessor *ja  = find_attr(prim, cgltf_attribute_type_joints, 0);
            cgltf_accessor *wa  = find_attr(prim, cgltf_attribute_type_weights, 0);
            if (!pos || !ja || !wa) continue;            /* skip non-skinned primitives */
            /* marrow's vertex model is 4 influences (DemoVertex.bones[4]); a mesh with JOINTS_1+ would
             * be silently truncated to wrong skinning, which validation can't catch (it compares the
             * same truncated data) - reject it outright. */
            if (find_attr(prim, cgltf_attribute_type_joints, 1)) {
                fprintf(stderr, "[assets_gltf] mesh has >4 joint influences (JOINTS_1+); marrow supports 4 per vertex\n");
                return -1;
            }
            total_v += pos->count;
            total_i += prim->indices ? prim->indices->count : pos->count;
        }
    }
    if (total_v == 0) { fprintf(stderr, "[assets_gltf] no skinned triangle mesh bound to the skin\n"); return -1; }
    if (total_v > 0xFFFFFFFFu || total_i > 0xFFFFFFFFu) { fprintf(stderr, "[assets_gltf] mesh too large for 32-bit indices\n"); return -1; }

    DemoVertex *verts = (DemoVertex *)malloc((size_t)total_v * sizeof(DemoVertex));
    uint32_t   *idx   = (uint32_t   *)malloc((size_t)total_i * sizeof(uint32_t));
    if (!verts || !idx) { free(verts); free(idx); fprintf(stderr, "[assets_gltf] out of memory building mesh\n"); return -1; }

    /* pass 2: fill (identical traversal order) */
    uint32_t vbase = 0, ii = 0;
    for (cgltf_size ni = 0; ni < data->nodes_count; ++ni) {
        const cgltf_node *node = &data->nodes[ni];
        if (node->skin != skin || !node->mesh) continue;
        for (cgltf_size pi = 0; pi < node->mesh->primitives_count; ++pi) {
            const cgltf_primitive *prim = &node->mesh->primitives[pi];
            if (prim->type != cgltf_primitive_type_triangles) continue;
            cgltf_accessor *pos = find_attr(prim, cgltf_attribute_type_position, 0);
            cgltf_accessor *ja  = find_attr(prim, cgltf_attribute_type_joints, 0);
            cgltf_accessor *wa  = find_attr(prim, cgltf_attribute_type_weights, 0);
            if (!pos || !ja || !wa) continue;
            cgltf_accessor *na = find_attr(prim, cgltf_attribute_type_normal, 0);
            cgltf_accessor *ta = find_attr(prim, cgltf_attribute_type_tangent, 0);

            uint32_t vc = (uint32_t)pos->count;
            for (uint32_t v = 0; v < vc; ++v) {
                DemoVertex *dv = &verts[vbase + v];
                /* Reject non-finite attribute data: cgltf_validate checks structure, not float
                 * payloads, and a NaN/Inf would silently poison both the render and the CPU-vs-GPU
                 * skinning comparison. */
                float P[3];
                if (!cgltf_accessor_read_float(pos, v, P, 3)) goto read_err;
                if (!all_finite(P, 3)) { fprintf(stderr, "[assets_gltf] non-finite POSITION\n"); goto fail; }
                dv->pos[0] = P[0]; dv->pos[1] = P[1]; dv->pos[2] = P[2];

                if (na) { float N[3]; if (!cgltf_accessor_read_float(na, v, N, 3)) goto read_err;
                          if (!all_finite(N, 3)) { fprintf(stderr, "[assets_gltf] non-finite NORMAL\n"); goto fail; }
                          dv->nrm[0] = N[0]; dv->nrm[1] = N[1]; dv->nrm[2] = N[2]; }
                else    { dv->nrm[0] = 0.0f; dv->nrm[1] = 1.0f; dv->nrm[2] = 0.0f; }

                if (ta) { float T[4]; if (!cgltf_accessor_read_float(ta, v, T, 4)) goto read_err;
                          if (!all_finite(T, 4)) { fprintf(stderr, "[assets_gltf] non-finite TANGENT\n"); goto fail; }
                          dv->tan[0] = T[0]; dv->tan[1] = T[1]; dv->tan[2] = T[2]; dv->tan[3] = T[3]; }
                else    { dv->tan[0] = 1.0f; dv->tan[1] = 0.0f; dv->tan[2] = 0.0f; dv->tan[3] = 1.0f; }

                cgltf_uint J[4] = { 0, 0, 0, 0 }; float W[4] = { 0, 0, 0, 0 };
                if (!cgltf_accessor_read_uint(ja, v, J, 4) || !cgltf_accessor_read_float(wa, v, W, 4)) goto read_err;
                if (!all_finite(W, 4)) { fprintf(stderr, "[assets_gltf] non-finite WEIGHTS_0\n"); goto fail; }
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    float w = W[k] > 0.0f ? W[k] : 0.0f;
                    if (w > 0.0f && J[k] >= jc_skin) {
                        fprintf(stderr, "[assets_gltf] JOINTS_0 index %u out of range (skin has %u joints)\n", (unsigned)J[k], jc_skin);
                        goto fail;
                    }
                    dv->bones[k]   = (J[k] < jc_skin) ? jo->marrow_of[J[k]] : 0u;
                    dv->weights[k] = w;
                    sum += w;
                }
                if (sum > 0.0f) { for (int k = 0; k < 4; ++k) dv->weights[k] /= sum; }
                else { dv->weights[0] = 1.0f; dv->weights[1] = dv->weights[2] = dv->weights[3] = 0.0f; }
            }

            uint32_t ic = prim->indices ? (uint32_t)prim->indices->count : vc;
            for (uint32_t e = 0; e < ic; ++e) {
                uint32_t local = prim->indices ? (uint32_t)cgltf_accessor_read_index(prim->indices, e) : e;
                if (local >= vc) {   /* keep vbase+local inside the uploaded vertex range (no GPU OOB) */
                    fprintf(stderr, "[assets_gltf] index %u out of range (primitive has %u verts)\n", local, vc);
                    goto fail;
                }
                idx[ii++] = vbase + local;
            }
            vbase += vc;
        }
    }

    out->verts   = verts;   out->vert_count  = (uint32_t)total_v;
    out->indices = idx;     out->index_count = (uint32_t)total_i;
    return 0;

read_err:
    fprintf(stderr, "[assets_gltf] mesh accessor read failed\n");
fail:
    free(verts); free(idx);
    return -1;
}

int assets_gltf_build(const char *mrw_path, const char *gltf_path, ProcAssets *out) {
    memset(out, 0, sizeof *out);
    int rc = 1;
    uint8_t *blob_buf = NULL; size_t blob_size = 0;
    cgltf_data *data = NULL;
    mrw_g2m_joint_order jo = {0};

    /* 1. load + validate the pre-baked .mrw, then reconstruct the demo clip table */
    if (read_file_aligned(mrw_path, &blob_buf, &blob_size)) goto done;
    mrw_blob blob;
    if (mrw_blob_open(blob_buf, (uint64_t)blob_size, &blob) != MRW_OK) {
        fprintf(stderr, "[assets_gltf] '%s' is not a valid .mrw blob\n", mrw_path); goto done;
    }
    mrw_skeleton_view sv;
    if (mrw_blob_skeleton(&blob, &sv) != MRW_OK) { fprintf(stderr, "[assets_gltf] .mrw has no skeleton\n"); goto done; }
    out->blob = blob_buf; out->blob_size = blob_size; blob_buf = NULL;   /* ownership -> out */
    out->joint_count = sv.joint_count;
    if (build_clip_table(&blob, out)) goto done;

    /* 2. parse the source glTF for the mesh + skin */
    cgltf_options copt = {0};
    if (cgltf_parse_file(&copt, gltf_path, &data) != cgltf_result_success) {
        fprintf(stderr, "[assets_gltf] cannot parse glTF '%s'\n", gltf_path); goto done;
    }
    if (cgltf_load_buffers(&copt, data, gltf_path) != cgltf_result_success) {
        fprintf(stderr, "[assets_gltf] cannot load glTF buffers for '%s'\n", gltf_path); goto done;
    }
    if (cgltf_validate(data) != cgltf_result_success) { fprintf(stderr, "[assets_gltf] glTF failed validation\n"); goto done; }
    if (data->skins_count == 0) { fprintf(stderr, "[assets_gltf] glTF has no skin\n"); goto done; }
    /* Refuse skin ambiguity, like gltf2marrow: with >1 skin we can't know which one the .mrw was
     * built from (a name/count cross-check can't disambiguate shared-skeleton skins). */
    if (data->skins_count > 1) {
        fprintf(stderr, "[assets_gltf] glTF has %zu skins; export a single-skin glTF for the demo\n", data->skins_count);
        goto done;
    }
    const cgltf_skin *skin = &data->skins[0];

    /* 3. derive the marrow joint order from the skin (same DFS gltf2marrow used) + sanity-check it
     * really matches the loaded skeleton (joint count, then names) before trusting the remap. */
    char jdiag[256] = {0};
    if (mrw_g2m_joint_order_build(skin, &jo, jdiag, sizeof jdiag) != MRW_OK) {
        fprintf(stderr, "[assets_gltf] joint order: %s\n", jdiag); goto done;
    }
    if (jo.joint_count != out->joint_count) {
        fprintf(stderr, "[assets_gltf] joint count mismatch: glTF skin %u vs .mrw %u (wrong .mrw for this glTF?)\n",
                jo.joint_count, out->joint_count); goto done;
    }
    if (cross_check_names(&sv, skin, &jo)) goto done;

    /* 4. mesh geometry */
    if (build_mesh(data, skin, &jo, out)) goto done;

    rc = 0;
    fprintf(stderr, "[assets_gltf] '%s': %u joints, %u clips%s; mesh '%s': %u verts, %u indices\n",
            mrw_path, out->joint_count, out->clip_count, out->tier_b_eligible ? " (baked)" : " (Tier-A only)",
            gltf_path, out->vert_count, out->index_count);

done:
    mrw_g2m_joint_order_free(&jo);                 /* zeroed when never built */
    if (data) cgltf_free(data);
    if (blob_buf) mrw_authoring_free(blob_buf);    /* only when not transferred to out */
    if (rc) {                                       /* failure: release whatever reached out */
        if (out->blob) mrw_authoring_free(out->blob);
        free(out->verts); free(out->indices);
        memset(out, 0, sizeof *out);
    }
    return rc;
}
