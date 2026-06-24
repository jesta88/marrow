/* --mesh probe extraction for marrow-bake (front-end TU; isolates the cgltf dependency). For each
 * marrow joint, the probes are the 8 corners of the AABB of the mesh vertices it actually influences
 * (across every JOINTS_n/WEIGHTS_n set, weight > 1e-3). The mesh's skin-order joint index maps to a
 * marrow joint by NODE NAME against the .mrw skeleton's name table - the only decoupled way to relate
 * the two without re-deriving gltf2marrow's topological remap. The map MUST be a complete, unique
 * bijection for the selected skin; a missing / duplicate / unnamed joint is an error by default
 * (silently swapping in synthetic probes could certify the wrong rig) and a per-bone box only under
 * --allow-probe-fallback. A mapped-but-uninfluenced bone gets 0 probes (eligible by default).
 * glTF skinning ignores the mesh node's own transform, so POSITION values are already the
 * bind-space probe points the residual test expects. */
#include "bake_run.h"
#include "mrw_bake.h"
#include "mrw_authoring.h"
#include "cgltf.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MESH_WEIGHT_EPS 1.0e-3f
#define MAX_JW_SETS     8        /* glTF skinning sets per primitive we'll consume */

static void mp_diag(char *d, size_t cap, const char *fmt, ...) {
    if (!d || !cap) return;
    va_list ap; va_start(ap, fmt); vsnprintf(d, cap, fmt, ap); va_end(ap);
}

static cgltf_accessor *find_attr(const cgltf_primitive *p, cgltf_attribute_type ty, int index) {
    for (cgltf_size i = 0; i < p->attributes_count; ++i)
        if (p->attributes[i].type == ty && p->attributes[i].index == index) return p->attributes[i].data;
    return NULL;
}

/* Bind-pose model positions (for the optional --allow-probe-fallback box). parent[j] < j (topo). */
static void bind_model_positions(const mrw_skeleton_view *sv, uint32_t jc, float *model) {
    for (uint32_t j = 0; j < jc; ++j) {
        uint16_t parent = 0xFFFFu; mrw_skeleton_parent(sv, j, &parent);
        mrw_trs trs; mrw_skeleton_rest_local(sv, j, &trs);
        float local[12]; mrw_trs_to_affine(&trs, local);
        if (parent == 0xFFFFu) memcpy(model + (size_t)j*12, local, sizeof local);
        else mrw_affine_mul(model + (size_t)parent*12, local, model + (size_t)j*12);
    }
}

mrw_result mrw_bake_mesh_probes(const mrw_bake_options *opt, const mrw_skeleton_view *sv,
                                uint32_t *probe_counts, float **out_probes, char *diag, size_t cap) {
    uint32_t jc = sv->joint_count;
    mrw_result rc = MRW_OK;

    const char **mnames = NULL;
    int      *sj2m = NULL;        /* skin-joint index → marrow-joint index (−1 = unmapped) */
    uint8_t  *mmapped = NULL;     /* 1 if some skin joint maps to marrow joint j            */
    uint8_t  *inf = NULL;         /* 1 if marrow joint j received ≥1 influence              */
    float    *mn = NULL, *mx = NULL, *model = NULL, *probes = NULL;
    cgltf_data *data = NULL;

    mnames = (const char **)mrw_authoring_alloc((size_t)jc * sizeof(const char *));
    if (!mnames) { rc = MRW_E_OVERFLOW; mp_diag(diag, cap, "out of memory"); goto done; }
    for (uint32_t j = 0; j < jc; ++j) mnames[j] = NULL;
    for (uint32_t j = 0; j < jc; ++j)
        if ((rc = mrw_skeleton_joint_name(sv, j, &mnames[j]))) { mp_diag(diag, cap, "cannot read skeleton joint name"); goto done; }

    cgltf_options copt = {0};
    if (cgltf_parse_file(&copt, opt->mesh_path, &data) != cgltf_result_success) { rc = MRW_E_FORMAT; mp_diag(diag, cap, "--mesh: cannot parse glTF"); goto done; }
    if (cgltf_load_buffers(&copt, data, opt->mesh_path) != cgltf_result_success) { rc = MRW_E_FORMAT; mp_diag(diag, cap, "--mesh: cannot load glTF buffers"); goto done; }
    if (cgltf_validate(data) != cgltf_result_success) { rc = MRW_E_FORMAT; mp_diag(diag, cap, "--mesh: glTF failed validation"); goto done; }

    /* select skin */
    const cgltf_skin *skin = NULL;
    if (opt->mesh_skin_index >= 0) {
        if ((cgltf_size)opt->mesh_skin_index >= data->skins_count) { rc = MRW_E_RANGE; mp_diag(diag, cap, "--mesh-skin %d out of range (%zu skins)", opt->mesh_skin_index, data->skins_count); goto done; }
        skin = &data->skins[opt->mesh_skin_index];
    } else if (data->skins_count == 1) {
        skin = &data->skins[0];
    } else { rc = MRW_E_FORMAT; mp_diag(diag, cap, "--mesh has %zu skins; pass --mesh-skin <index>", data->skins_count); goto done; }

    uint32_t nsj = (uint32_t)skin->joints_count;
    sj2m    = (int *)mrw_authoring_alloc((size_t)(nsj ? nsj : 1) * sizeof(int));
    mmapped = (uint8_t *)mrw_authoring_alloc((size_t)jc);
    inf     = (uint8_t *)mrw_authoring_alloc((size_t)jc);
    mn      = (float *)mrw_authoring_alloc((size_t)jc * 3 * sizeof(float));
    mx      = (float *)mrw_authoring_alloc((size_t)jc * 3 * sizeof(float));
    model   = (float *)mrw_authoring_alloc((size_t)jc * 12 * sizeof(float));
    if (!sj2m || !mmapped || !inf || !mn || !mx || !model) { rc = MRW_E_OVERFLOW; mp_diag(diag, cap, "out of memory"); goto done; }
    memset(mmapped, 0, (size_t)jc);
    memset(inf, 0, (size_t)jc);

    /* name map: skin joint → marrow joint (complete + unique). */
    int fallback = opt->allow_probe_fallback;
    for (uint32_t i = 0; i < nsj; ++i) {
        const cgltf_node *jn = skin->joints[i];
        const char *nm = jn ? jn->name : NULL;
        int found = -1;
        if (nm && nm[0]) for (uint32_t j = 0; j < jc; ++j) if (mnames[j] && strcmp(mnames[j], nm) == 0) { found = (int)j; break; }
        if (found < 0) {
            if (!fallback) { rc = MRW_E_INCOMPATIBLE; mp_diag(diag, cap, "--mesh skin joint %u (%s) has no matching skeleton joint", i, (nm && nm[0]) ? nm : "<unnamed>"); goto done; }
            sj2m[i] = -1; continue;
        }
        if (mmapped[found]) {
            if (!fallback) { rc = MRW_E_INCOMPATIBLE; mp_diag(diag, cap, "--mesh: duplicate joint name '%s' maps to one skeleton joint", nm); goto done; }
            sj2m[i] = -1; continue;
        }
        sj2m[i] = found; mmapped[found] = 1;
    }
    /* completeness: every marrow joint must be covered (else the rig isn't this mesh's rig). */
    for (uint32_t j = 0; j < jc; ++j)
        if (!mmapped[j] && !fallback) { rc = MRW_E_INCOMPATIBLE; mp_diag(diag, cap, "--mesh: skeleton joint '%s' has no corresponding skin joint", mnames[j] ? mnames[j] : "?"); goto done; }

    /* accumulate per-bone influence AABBs over every skinned primitive bound to this skin. */
    for (cgltf_size ni = 0; ni < data->nodes_count; ++ni) {
        const cgltf_node *node = &data->nodes[ni];
        if (node->skin != skin || !node->mesh) continue;
        const cgltf_mesh *mesh = node->mesh;
        for (cgltf_size pi = 0; pi < mesh->primitives_count; ++pi) {
            const cgltf_primitive *prim = &mesh->primitives[pi];
            cgltf_accessor *pos = find_attr(prim, cgltf_attribute_type_position, 0);
            if (!pos) continue;

            /* Gather EVERY present JOINTS_n/WEIGHTS_n pair (not just contiguous from 0): find the max
             * set index and pair each present set, so a file with e.g. JOINTS_0 + JOINTS_2 keeps set 2. */
            int maxset = -1;
            for (cgltf_size ai = 0; ai < prim->attributes_count; ++ai) {
                const cgltf_attribute *a = &prim->attributes[ai];
                if ((a->type == cgltf_attribute_type_joints || a->type == cgltf_attribute_type_weights) && a->index > maxset)
                    maxset = a->index;
            }
            if (maxset >= MAX_JW_SETS) { rc = MRW_E_UNSUPPORTED; mp_diag(diag, cap, "--mesh: too many JOINTS/WEIGHTS sets (%d)", maxset + 1); goto done; }
            cgltf_accessor *jacc[MAX_JW_SETS], *wacc[MAX_JW_SETS]; int nset = 0;
            for (int set = 0; set <= maxset; ++set) {
                cgltf_accessor *ja = find_attr(prim, cgltf_attribute_type_joints, set);
                cgltf_accessor *wa = find_attr(prim, cgltf_attribute_type_weights, set);
                if (!ja && !wa) continue;                       /* gap (non-contiguous) - skip cleanly */
                if (!ja || !wa) { rc = MRW_E_FORMAT; mp_diag(diag, cap, "--mesh: JOINTS_%d/WEIGHTS_%d not paired", set, set); goto done; }
                jacc[nset] = ja; wacc[nset] = wa; ++nset;
            }
            if (nset == 0) continue;                            /* a non-skinned primitive */

            uint32_t vc = (uint32_t)pos->count;
            for (uint32_t v = 0; v < vc; ++v) {
                float P[3];
                if (!cgltf_accessor_read_float(pos, v, P, 3)) { rc = MRW_E_FORMAT; mp_diag(diag, cap, "--mesh: POSITION read failed"); goto done; }
                for (int si = 0; si < nset; ++si) {
                    cgltf_uint J[4]; float W[4];
                    if (!cgltf_accessor_read_uint(jacc[si], v, J, 4) || !cgltf_accessor_read_float(wacc[si], v, W, 4)) { rc = MRW_E_FORMAT; mp_diag(diag, cap, "--mesh: JOINTS/WEIGHTS read failed"); goto done; }
                    for (int k = 0; k < 4; ++k) {
                        if (W[k] <= MESH_WEIGHT_EPS) continue;
                        if (J[k] >= nsj) { rc = MRW_E_FORMAT; mp_diag(diag, cap, "--mesh: joint index %u out of range (skin has %u joints)", (unsigned)J[k], nsj); goto done; }
                        int mj = sj2m[J[k]];
                        if (mj < 0) continue;          /* dropped (fallback) skin joint */
                        if (!inf[mj]) { for (int c = 0; c < 3; ++c) mn[mj*3+c] = mx[mj*3+c] = P[c]; inf[mj] = 1; }
                        else for (int c = 0; c < 3; ++c) { if (P[c] < mn[mj*3+c]) mn[mj*3+c] = P[c]; if (P[c] > mx[mj*3+c]) mx[mj*3+c] = P[c]; }
                    }
                }
            }
        }
    }

    /* per-bone probe counts: 8 for an influenced bone, 8 box corners for a fallback-unmapped bone,
     * 0 otherwise (mapped-but-uninfluenced ⇒ eligible by default). */
    int need_box = 0;
    uint64_t total = 0;
    for (uint32_t j = 0; j < jc; ++j) {
        int has = inf[j] || (fallback && !mmapped[j]);
        probe_counts[j] = has ? 8u : 0u;
        if (fallback && !mmapped[j] && !inf[j]) need_box = 1;
        total += probe_counts[j];
    }
    /* A --mesh that yields no probes at all (no skinned primitive bound to the skin, or every bone
     * uninfluenced) would silently certify the rig with no perceptual test - reject it instead so the
     * caller drops --mesh (box probes) or fixes the mesh, rather than getting a vacuous pass. */
    if (total == 0) { rc = MRW_E_INCOMPATIBLE; mp_diag(diag, cap, "--mesh produced no probes (no skinned influences for the selected skin)"); goto done; }

    if (need_box) bind_model_positions(sv, jc, model);
    float r = (opt->probe_radius > 0.0f) ? opt->probe_radius : 0.05f;

    probes = (float *)mrw_authoring_alloc((size_t)(total ? total : 1) * 3 * sizeof(float));
    if (!probes) { rc = MRW_E_OVERFLOW; mp_diag(diag, cap, "out of memory"); goto done; }
    {
        size_t off = 0;
        for (uint32_t j = 0; j < jc; ++j) {
            if (probe_counts[j] == 0) continue;
            float lo[3], hi[3];
            if (inf[j]) { for (int c = 0; c < 3; ++c) { lo[c] = mn[j*3+c]; hi[c] = mx[j*3+c]; } }
            else { /* fallback box around the bind-pose position */
                float bx=model[j*12+3], by=model[j*12+7], bz=model[j*12+11];
                lo[0]=bx-r; lo[1]=by-r; lo[2]=bz-r; hi[0]=bx+r; hi[1]=by+r; hi[2]=bz+r;
            }
            for (int sx=0;sx<2;++sx) for (int sy=0;sy<2;++sy) for (int sz=0;sz<2;++sz) {
                probes[off++] = sx ? hi[0] : lo[0];
                probes[off++] = sy ? hi[1] : lo[1];
                probes[off++] = sz ? hi[2] : lo[2];
            }
        }
    }

    *out_probes = probes; probes = NULL;   /* ownership transferred */
    rc = MRW_OK;

done:
    mrw_authoring_free(probes);
    mrw_authoring_free(model);
    mrw_authoring_free(mx);
    mrw_authoring_free(mn);
    mrw_authoring_free(inf);
    mrw_authoring_free(mmapped);
    mrw_authoring_free(sj2m);
    mrw_authoring_free(mnames);
    if (data) cgltf_free(data);
    return rc;
}
