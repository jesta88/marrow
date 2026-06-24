/* marrow-bake front-end. See bake_run.h for the contract. Flow: read the input .mrw → validate it is
 * a CPU clip set (one SKELETON, ≥1 CLIP, no BAKED) → reconstruct authoring structs from the
 * validated views (typed copies, never a cast over blob bytes) → build probe points →
 * choose per-clip frame counts → bake every clip (eligibility AND-accumulated) →
 * re-serialize through the checked writer (BAKED when eligible, clip-set-only otherwise) →
 * self-validate. cgltf is used only on the --mesh path. */
#include "bake_run.h"
#include "mrw_bake.h"
#include "mrw_authoring.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* A per-clip baked stream of more than this many frames is almost certainly a bad --bake-fps; reject
 * before the cast/allocation rather than attempt a multi-GB bake. */
#define MRW_BAKE_MAX_FRAMES (1u << 24)

static const char *bake_reason_str(mrw_bake_reason r) {
    switch (r) {
        case MRW_BAKE_STRUCTURAL: return "non-decomposable transform";
        case MRW_BAKE_RESIDUAL:   return "residual exceeds tolerance";
        case MRW_BAKE_QUANTIZED:  return "texel overflow (non-finite)";
        default:                  return "ok";
    }
}

static void set_diag(char *diag, size_t cap, const char *msg) {
    if (diag && cap) { snprintf(diag, cap, "%s", msg); }
}

/* Read a whole file into a fresh 64-aligned buffer (mrw_blob_open requires ≥64 alignment). */
static mrw_result read_file(const char *path, uint8_t **out_buf, size_t *out_size,
                            char *diag, size_t cap) {
    *out_buf = NULL; *out_size = 0;
    FILE *f = fopen(path, "rb");
    if (!f) { set_diag(diag, cap, "cannot open input file"); return MRW_E_FORMAT; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); set_diag(diag, cap, "cannot seek input file"); return MRW_E_FORMAT; }
    long n = ftell(f);
    if (n <= 0) { fclose(f); set_diag(diag, cap, "input file is empty or unreadable"); return MRW_E_FORMAT; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); set_diag(diag, cap, "cannot rewind input file"); return MRW_E_FORMAT; }
    uint8_t *buf = (uint8_t *)mrw_authoring_alloc((size_t)n);
    if (!buf) { fclose(f); set_diag(diag, cap, "out of memory reading input"); return MRW_E_OVERFLOW; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { mrw_authoring_free(buf); set_diag(diag, cap, "short read on input file"); return MRW_E_FORMAT; }
    *out_buf = buf; *out_size = (size_t)n;
    return MRW_OK;
}

/* Default (no-mesh) probes: 8 corners of a ±r box at each bone's bind-pose model position. Composes
 * rest_local down the hierarchy (parent[j] < j, topo order - guaranteed by the loader). Allocates
 * *out_probes (jc·8·3 floats) and fills probe_counts[j] = 8. */
static mrw_result build_box_probes(uint32_t jc, const uint16_t *parent, const float *rest, float r,
                                   uint32_t *probe_counts, float **out_probes) {
    float *model  = (float *)mrw_authoring_alloc((size_t)jc * 12 * sizeof(float));
    float *probes = (float *)mrw_authoring_alloc((size_t)jc * 8 * 3 * sizeof(float));
    if (!model || !probes) { mrw_authoring_free(model); mrw_authoring_free(probes); return MRW_E_OVERFLOW; }
    for (uint32_t j = 0; j < jc; ++j) {
        mrw_trs trs;
        memcpy(trs.rot,   rest + j*10 + 0, 16);
        memcpy(trs.trans, rest + j*10 + 4, 12);
        memcpy(trs.scale, rest + j*10 + 7, 12);
        float local[12]; mrw_trs_to_affine(&trs, local);
        if (parent[j] == 0xFFFFu) memcpy(model + (size_t)j*12, local, sizeof local);
        else mrw_affine_mul(model + (size_t)parent[j]*12, local, model + (size_t)j*12);
        float bx = model[j*12+3], by = model[j*12+7], bz = model[j*12+11];
        uint32_t k = 0;
        for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2)
        for (int sz = -1; sz <= 1; sz += 2) {
            float *p = probes + ((size_t)j*8 + k)*3;
            p[0] = bx + (float)sx*r; p[1] = by + (float)sy*r; p[2] = bz + (float)sz*r; ++k;
        }
        probe_counts[j] = 8;
    }
    mrw_authoring_free(model);
    *out_probes = probes;
    return MRW_OK;
}

/* --mesh probe extraction (real cgltf path). Implemented in mesh_probes.c (front-end TU); declared
 * here to keep the cgltf dependency out of this file. */
mrw_result mrw_bake_mesh_probes(const mrw_bake_options *opt, const mrw_skeleton_view *sv,
                                uint32_t *probe_counts, float **out_probes, char *diag, size_t cap);

/* Diagonal of the AABB of `npts` probe points - the model extent used for the default tolerance. */
static float probe_aabb_diag(const float *probes, size_t npts) {
    if (npts == 0) return 0.0f;
    float mn[3], mx[3];
    for (int k = 0; k < 3; ++k) { mn[k] = mx[k] = probes[k]; }
    for (size_t i = 0; i < npts; ++i)
        for (int k = 0; k < 3; ++k) {
            float v = probes[i*3 + k];
            if (v < mn[k]) mn[k] = v;
            if (v > mx[k]) mx[k] = v;
        }
    float dx = mx[0]-mn[0], dy = mx[1]-mn[1], dz = mx[2]-mn[2];
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

mrw_result mrw_bake_run(const mrw_bake_options *opt, uint8_t **out_buf, size_t *out_size,
                        int *out_eligible, mrw_bake_stats *out_worst, char *diag, size_t diag_cap) {
    if (out_buf) *out_buf = NULL;
    if (!opt || !opt->input_path || !out_buf || !out_size) { set_diag(diag, diag_cap, "bad arguments"); return MRW_E_FORMAT; }
    if (!(opt->bake_fps > 0.0f) || !isfinite(opt->bake_fps)) { set_diag(diag, diag_cap, "--bake-fps must be finite and > 0"); return MRW_E_RANGE; }

    /* Everything below is freed at `cleanup`; *out_buf is transferred out on success (outbuf NULLed). */
    uint32_t   nclip = 0;          /* read again in cleanup (guarded by clips != NULL) */
    uint8_t   *inbuf = NULL;        size_t insize = 0;
    mrw_clip_view *clipv = NULL;
    uint16_t  *parent = NULL;
    float     *rest = NULL, *ib = NULL;
    const char **names = NULL;
    mrw_clip  *clips = NULL;
    uint32_t  *probe_counts = NULL; float *probes = NULL;
    uint32_t  *first_frame = NULL, *frame_count = NULL, *clip_flags = NULL, *clip_index = NULL;
    float     *source_dur = NULL;
    void      *bake_scratch = NULL; uint16_t *texels = NULL;
    uint8_t   *outbuf = NULL;       size_t outsize = 0;
    mrw_result rc;

    rc = read_file(opt->input_path, &inbuf, &insize, diag, diag_cap);
    if (rc) return rc;

    mrw_blob blob;
    rc = mrw_blob_open(inbuf, (uint64_t)insize, &blob);
    if (rc) { set_diag(diag, diag_cap, "input is not a valid .mrw blob"); goto cleanup; }

    /* Require a CPU clip set: exactly one SKELETON, ≥1 CLIP, no pre-existing BAKED. */
    uint32_t n_skel = 0, n_baked = 0;
    for (uint32_t i = 0; i < blob.section_count; ++i) {
        uint32_t ty = 0;
        if (mrw_blob_section_type(&blob, i, &ty) != MRW_OK) continue;
        if (ty == MRW_SECTION_SKELETON) ++n_skel;
        else if (ty == MRW_SECTION_CLIP) ++nclip;
        else if (ty == MRW_SECTION_BAKED) ++n_baked;
    }
    if (n_skel != 1) { rc = MRW_E_INCOMPATIBLE; set_diag(diag, diag_cap, "input must contain exactly one SKELETON"); goto cleanup; }
    if (nclip < 1)   { rc = MRW_E_INCOMPATIBLE; set_diag(diag, diag_cap, "input must contain at least one CLIP"); goto cleanup; }
    if (n_baked != 0){ rc = MRW_E_INCOMPATIBLE; set_diag(diag, diag_cap, "input already contains a BAKED section (expected a Tier-A clip set)"); goto cleanup; }

    mrw_skeleton_view sv;
    rc = mrw_blob_skeleton(&blob, &sv);
    if (rc) { set_diag(diag, diag_cap, "cannot read SKELETON"); goto cleanup; }
    uint32_t jc = sv.joint_count;

    /* Collect clip views in section order. */
    clipv = (mrw_clip_view *)mrw_authoring_alloc((size_t)nclip * sizeof(*clipv));
    if (!clipv) { rc = MRW_E_OVERFLOW; set_diag(diag, diag_cap, "out of memory"); goto cleanup; }
    { uint32_t ci = 0;
      for (uint32_t i = 0; i < blob.section_count; ++i) {
          uint32_t ty = 0;
          if (mrw_blob_section_type(&blob, i, &ty) != MRW_OK || ty != MRW_SECTION_CLIP) continue;
          rc = mrw_clip_view_at(&blob, i, &clipv[ci]);
          if (rc) { set_diag(diag, diag_cap, "cannot read CLIP"); goto cleanup; }
          if (clipv[ci].joint_count != jc) { rc = MRW_E_INCOMPATIBLE; set_diag(diag, diag_cap, "clip joint count does not match skeleton"); goto cleanup; }
          ++ci;
      }
    }

    /* Reconstruct the skeleton struct (typed copies; names borrow `inbuf`, kept alive until build). */
    parent = (uint16_t *)mrw_authoring_alloc((size_t)jc * sizeof(uint16_t));
    rest   = (float *)mrw_authoring_alloc((size_t)jc * 10 * sizeof(float));
    ib     = (float *)mrw_authoring_alloc((size_t)jc * 12 * sizeof(float));
    names  = (const char **)mrw_authoring_alloc((size_t)jc * sizeof(const char *));
    if (!parent || !rest || !ib || !names) { rc = MRW_E_OVERFLOW; set_diag(diag, diag_cap, "out of memory"); goto cleanup; }
    for (uint32_t j = 0; j < jc; ++j) {
        if ((rc = mrw_skeleton_parent(&sv, j, &parent[j]))) { set_diag(diag, diag_cap, "cannot read parent"); goto cleanup; }
        mrw_trs trs;
        if ((rc = mrw_skeleton_rest_local(&sv, j, &trs))) { set_diag(diag, diag_cap, "cannot read rest_local"); goto cleanup; }
        memcpy(rest + (size_t)j*10, &trs, 40);   /* mrw_trs == q4,t3,s3 == wire rest_local */
        if ((rc = mrw_skeleton_inverse_bind(&sv, j, ib + (size_t)j*12))) { set_diag(diag, diag_cap, "cannot read inverse_bind"); goto cleanup; }
        if ((rc = mrw_skeleton_joint_name(&sv, j, &names[j]))) { set_diag(diag, diag_cap, "cannot read joint name"); goto cleanup; }
    }
    mrw_skel skel = { jc, parent, rest, ib, names };

    /* Reconstruct each clip (typed copies of samples / optional root_track). */
    clips = (mrw_clip *)mrw_authoring_alloc((size_t)nclip * sizeof(*clips));
    if (!clips) { rc = MRW_E_OVERFLOW; set_diag(diag, diag_cap, "out of memory"); goto cleanup; }
    memset(clips, 0, (size_t)nclip * sizeof(*clips));
    for (uint32_t c = 0; c < nclip; ++c) {
        const mrw_clip_view *cv = &clipv[c];
        uint32_t sc = cv->sample_count;
        float *samples = (float *)mrw_authoring_alloc((size_t)jc * sc * 10 * sizeof(float));
        if (!samples) { rc = MRW_E_OVERFLOW; set_diag(diag, diag_cap, "out of memory"); goto cleanup; }
        for (uint32_t j = 0; j < jc; ++j)
            for (uint32_t s = 0; s < sc; ++s) {
                mrw_trs trs;
                if ((rc = mrw_clip_sample(cv, j, s, &trs))) { mrw_authoring_free(samples); set_diag(diag, diag_cap, "cannot read clip sample"); goto cleanup; }
                memcpy(samples + ((size_t)j*sc + s)*10, &trs, 40);
            }
        float *root = NULL;
        if (cv->flags & MRW_CLIP_HAS_ROOT_MOTION) {
            root = (float *)mrw_authoring_alloc((size_t)sc * 7 * sizeof(float));
            if (!root) { mrw_authoring_free(samples); rc = MRW_E_OVERFLOW; set_diag(diag, diag_cap, "out of memory"); goto cleanup; }
            for (uint32_t s = 0; s < sc; ++s) {
                mrw_xform rx;
                if ((rc = mrw_clip_root_sample(cv, s, &rx))) { mrw_authoring_free(samples); mrw_authoring_free(root); set_diag(diag, diag_cap, "cannot read root sample"); goto cleanup; }
                memcpy(root + (size_t)s*7 + 0, rx.rot,   16);
                memcpy(root + (size_t)s*7 + 4, rx.trans, 12);
            }
        }
        clips[c].fps = cv->fps; clips[c].sample_count = sc; clips[c].flags = cv->flags;
        clips[c].samples = samples; clips[c].root_track = root;
    }

    /* Probe points. */
    probe_counts = (uint32_t *)mrw_authoring_alloc((size_t)jc * sizeof(uint32_t));
    if (!probe_counts) { rc = MRW_E_OVERFLOW; set_diag(diag, diag_cap, "out of memory"); goto cleanup; }
    if (opt->mesh_path) {
        rc = mrw_bake_mesh_probes(opt, &sv, probe_counts, &probes, diag, diag_cap);
        if (rc) goto cleanup;            /* mesh path sets its own diag */
    } else {
        float r = (opt->probe_radius > 0.0f) ? opt->probe_radius : 0.05f;
        rc = build_box_probes(jc, parent, rest, r, probe_counts, &probes);
        if (rc) { set_diag(diag, diag_cap, "out of memory building probes"); goto cleanup; }
    }
    size_t total_probes = 0;
    for (uint32_t j = 0; j < jc; ++j) total_probes += probe_counts[j];

    /* Tolerance: explicit, or the default max(1 mm, 0.2% · model-AABB diagonal). */
    float tol = opt->decompose_tol;
    if (!(tol > 0.0f)) {
        float d = probe_aabb_diag(probes, total_probes);
        float t = 0.002f * d;
        tol = (t > 0.001f) ? t : 0.001f;
    }

    /* Per-clip frame counts (source_duration == 0 iff frame_count == 1). */
    first_frame = (uint32_t *)mrw_authoring_alloc((size_t)nclip * sizeof(uint32_t));
    frame_count = (uint32_t *)mrw_authoring_alloc((size_t)nclip * sizeof(uint32_t));
    clip_flags  = (uint32_t *)mrw_authoring_alloc((size_t)nclip * sizeof(uint32_t));
    clip_index  = (uint32_t *)mrw_authoring_alloc((size_t)nclip * sizeof(uint32_t));
    source_dur  = (float *)mrw_authoring_alloc((size_t)nclip * sizeof(float));
    if (!first_frame || !frame_count || !clip_flags || !clip_index || !source_dur) {
        rc = MRW_E_OVERFLOW; set_diag(diag, diag_cap, "out of memory"); goto cleanup;
    }
    uint64_t total_frames = 0;
    uint64_t fst = (uint64_t)jc * 2;
    for (uint32_t c = 0; c < nclip; ++c) {
        uint32_t sc = clips[c].sample_count;
        float dur = (sc <= 1) ? 0.0f : (float)(sc - 1) / clips[c].fps;  /* MUST match loader's clip_dur exactly */
        uint32_t fc;
        if (dur == 0.0f) fc = 1;
        else {
            double frames = (double)dur * (double)opt->bake_fps;
            if (!isfinite(frames) || frames > (double)MRW_BAKE_MAX_FRAMES) {
                rc = MRW_E_RANGE; set_diag(diag, diag_cap, "--bake-fps yields too many frames"); goto cleanup;
            }
            uint32_t n = (uint32_t)(frames + 0.5);   /* round-to-nearest, range-checked above */
            fc = n + 1;
            if (fc < 2) fc = 2;                       /* max(2, …): a dynamic clip bakes ≥ 2 frames */
        }
        first_frame[c] = (uint32_t)total_frames;
        frame_count[c] = fc;
        source_dur[c]  = dur;
        clip_flags[c]  = (clips[c].flags & MRW_CLIP_LOOPING) ? MRW_BAKED_CLIP_LOOPING : 0u;
        clip_index[c]  = c;
        total_frames  += fc;
        if (total_frames * fst > UINT32_MAX) {       /* texel_count is a u32 wire field */
            rc = MRW_E_OVERFLOW; set_diag(diag, diag_cap, "baked texel stream too large"); goto cleanup;
        }
    }

    /* Rig-wide texel buffer + reusable bake scratch (scratch depends only on joint_count). */
    mrw_mem_req sreq, treq_all;
    rc = mrw_bake_clip_requirements(jc, (uint32_t)total_frames, &sreq, &treq_all);
    if (rc) { set_diag(diag, diag_cap, "baked size overflow"); goto cleanup; }
    texels       = (uint16_t *)mrw_authoring_alloc(treq_all.size);  /* 64-aligned ⇒ ≥16-aligned */
    bake_scratch = mrw_authoring_alloc(sreq.size);
    if (!texels || !bake_scratch) { rc = MRW_E_OVERFLOW; set_diag(diag, diag_cap, "out of memory"); goto cleanup; }

    /* Bake every clip; eligibility is the logical AND across the whole clip set. */
    int rig_eligible = 1;
    mrw_bake_stats worst; worst.eligible = 1; worst.max_residual = -1.0f;
    worst.worst_bone = 0; worst.worst_frame = 0; worst.reason = MRW_BAKE_OK;
    uint32_t worst_clip = 0;
    for (uint32_t c = 0; c < nclip; ++c) {
        uint32_t fc = frame_count[c];
        uint16_t *base = texels + (size_t)first_frame[c] * fst * 4;
        mrw_mem_req treq_c;
        mrw_bake_clip_requirements(jc, fc, NULL, &treq_c);    /* re-validated, never errors here */
        mrw_bake_stats st;
        rc = mrw_bake_clip(&sv, &clipv[c], fc, probe_counts, probes, tol,
                           bake_scratch, sreq.size, base, treq_c.size, &st);
        if (rc) { set_diag(diag, diag_cap, "bake failed (Tier-A sampling)"); goto cleanup; }
        rig_eligible &= st.eligible;
        if (st.max_residual > worst.max_residual) { worst = st; worst_clip = c; }
    }

    /* Assemble + serialize: BAKED when eligible, clip-set-only otherwise. */
    if (rig_eligible) {
        mrw_baked baked;
        baked.frame_stride_texels = 0;                   /* ⇒ bone_count·2 (canonical tight pack) */
        baked.total_frames        = (uint32_t)total_frames;
        baked.texels              = texels;
        baked.clip_count          = nclip;
        baked.clip_index          = clip_index;
        baked.first_frame         = first_frame;
        baked.frame_count         = frame_count;
        baked.source_duration     = source_dur;
        baked.clip_flags          = clip_flags;
        rc = mrw_authoring_build(&skel, clips, nclip, &baked, &outbuf, &outsize);
    } else {
        rc = mrw_authoring_build(&skel, clips, nclip, NULL, &outbuf, &outsize);
    }
    if (rc) { set_diag(diag, diag_cap, "serialization failed"); goto cleanup; }

    /* Self-validate: never emit a malformed .mrw. */
    {
        mrw_blob chk;
        rc = mrw_blob_open(outbuf, (uint64_t)outsize, &chk);
        if (rc) { set_diag(diag, diag_cap, "self-validation failed (blob did not re-open)"); goto cleanup; }
        if (rig_eligible) {
            mrw_baked_view bv; int found = 0;
            for (uint32_t i = 0; i < chk.section_count; ++i) {
                uint32_t ty = 0;
                if (mrw_blob_section_type(&chk, i, &ty) == MRW_OK && ty == MRW_SECTION_BAKED) {
                    rc = mrw_baked_view_at(&chk, i, &bv);
                    if (rc) { set_diag(diag, diag_cap, "self-validation failed (BAKED view)"); goto cleanup; }
                    found = 1; break;
                }
            }
            if (!found) { rc = MRW_E_FORMAT; set_diag(diag, diag_cap, "self-validation failed (no BAKED section)"); goto cleanup; }
            for (uint32_t c = 0; c < nclip; ++c) {
                mrw_baked_clip ce;
                rc = mrw_baked_clip_entry(&bv, c, &ce);
                if (rc) { set_diag(diag, diag_cap, "self-validation failed (clip entry)"); goto cleanup; }
            }
        }
    }

    /* Success - transfer ownership and report. */
    *out_buf = outbuf; *out_size = outsize; outbuf = NULL;
    if (out_eligible) *out_eligible = rig_eligible;
    if (out_worst)    *out_worst = worst;
    if (diag && diag_cap) {
        if (rig_eligible)
            snprintf(diag, diag_cap, "baked %u clip(s) / %u frame(s); worst residual %.4g m (tol %.4g m)",
                     nclip, (uint32_t)total_frames, (double)worst.max_residual, (double)tol);
        else
            snprintf(diag, diag_cap, "rig ineligible for Tier B: clip %u bone %u frame %u: %s (residual %.4g m > tol %.4g m)",
                     worst_clip, worst.worst_bone, worst.worst_frame, bake_reason_str(worst.reason),
                     (double)worst.max_residual, (double)tol);
    }
    rc = MRW_OK;

cleanup:
    if (outbuf) mrw_authoring_free(outbuf);
    mrw_authoring_free(texels);
    mrw_authoring_free(bake_scratch);
    mrw_authoring_free(source_dur);
    mrw_authoring_free(clip_index);
    mrw_authoring_free(clip_flags);
    mrw_authoring_free(frame_count);
    mrw_authoring_free(first_frame);
    mrw_authoring_free(probes);
    mrw_authoring_free(probe_counts);
    if (clips) {
        for (uint32_t c = 0; c < nclip; ++c) {
            mrw_authoring_free((void *)clips[c].samples);
            mrw_authoring_free((void *)clips[c].root_track);
        }
        mrw_authoring_free(clips);
    }
    mrw_authoring_free(names);
    mrw_authoring_free(ib);
    mrw_authoring_free(rest);
    mrw_authoring_free(parent);
    mrw_authoring_free(clipv);
    mrw_authoring_free(inbuf);
    return rc;
}
