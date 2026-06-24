#include "validate.h"

#include "marrow.h"
#include "mrw_authoring.h"   /* mrw_authoring_alloc/free (64-aligned) */
#include "mrw_bake.h"        /* component-space reference: mrw_bake_sample_xform / mrw_xform_nlerp */
#include "profiler.h"        /* prof_now_s - the demo's single monotonic timer */
#include "jobs.h"            /* thread pool - the concurrency-parity gate fans the batch across cores */

#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#include "validate_skin_comp.spv.h"
#include "validate_skin_f16_comp.spv.h"
#include "validate_skin_f16_ssbo_comp.spv.h"

/* max basis abs-diff and max translation distance (mm) between two palettes (jc * 3x4). */
static void palette_diff(const float *A, const float *B, uint32_t jc, double *max_basis, double *max_trans_mm) {
    *max_basis = 0.0; *max_trans_mm = 0.0;
    for (uint32_t j = 0; j < jc; ++j) {
        const float *a = &A[j * 12], *b = &B[j * 12];
        static const int basis_idx[9] = { 0, 1, 2, 4, 5, 6, 8, 9, 10 };
        for (int k = 0; k < 9; ++k) {
            double d = fabs((double)a[basis_idx[k]] - (double)b[basis_idx[k]]);
            if (d > *max_basis) *max_basis = d;
        }
        double dx = a[3] - b[3], dy = a[7] - b[7], dz = a[11] - b[11];
        double tmm = sqrt(dx*dx + dy*dy + dz*dz) * 1000.0;
        if (tmm > *max_trans_mm) *max_trans_mm = tmm;
    }
}

static int find_baked(const mrw_blob *b, mrw_baked_view *out) {
    for (uint32_t i = 0; i < b->section_count; ++i) {
        uint32_t type = 0;
        if (mrw_blob_section_type(b, i, &type) == MRW_OK && type == MRW_SECTION_BAKED)
            return mrw_baked_view_at(b, i, out) == MRW_OK;
    }
    return 0;
}
static int find_clip(const ProcAssets *a, const char *name, const mrw_blob *blob,
                     mrw_clip_view *out, uint32_t *clip_index) {
    for (uint32_t i = 0; i < a->clip_count; ++i)
        if (strcmp(a->clips[i].name, name) == 0) {
            *clip_index = a->clips[i].clip_index;
            return mrw_clip_view_at(blob, 1 + a->clips[i].clip_index, out) == MRW_OK;
        }
    return 0;
}

int validate_run(const ProcAssets *assets, uint32_t bench_n) {
    mrw_blob blob;
    if (mrw_blob_open(assets->blob, assets->blob_size, &blob) != MRW_OK) { fprintf(stderr, "[validate] blob open failed\n"); return 1; }
    mrw_skeleton_view skel; mrw_blob_skeleton(&blob, &skel);
    mrw_baked_view bv; if (!find_baked(&blob, &bv)) { fprintf(stderr, "[validate] no BAKED section\n"); return 1; }
    mrw_clip_view walk, run; uint32_t walk_ci, run_ci;
    if (!find_clip(assets, "walk", &blob, &walk, &walk_ci) || !find_clip(assets, "run", &blob, &run, &run_ci)) {
        fprintf(stderr, "[validate] walk/run clip missing\n"); return 1;
    }
    mrw_baked_clip we, re;
    mrw_baked_clip_entry(&bv, walk_ci, &we);
    mrw_baked_clip_entry(&bv, run_ci, &re);
    const uint16_t *texels = (const uint16_t *)(bv.base + bv.texels_off);

    uint32_t jc = skel.joint_count;
    float *scratch  = (float *)mrw_authoring_alloc((size_t)jc * 12 * sizeof(float));
    float *paletteA = (float *)mrw_authoring_alloc((size_t)jc * 12 * sizeof(float));
    float *paletteB = (float *)mrw_authoring_alloc((size_t)jc * 12 * sizeof(float));
    mrw_trs *localsW = (mrw_trs *)mrw_authoring_alloc((size_t)jc * sizeof(mrw_trs));
    mrw_trs *localsR = (mrw_trs *)mrw_authoring_alloc((size_t)jc * sizeof(mrw_trs));
    float *model12  = (float *)mrw_authoring_alloc((size_t)jc * 12 * sizeof(float));

    int rc = 0;
    double mb, tmm;
    float dur = we.source_duration_s;

    fprintf(stderr, "\n=== marrow live validation ===\n");

    /* (1) single-clip EXACT-AT-FRAME: CPU vs baked at baked frame 1 -> sub-mm */
    {
        float t = (1.0f / (float)(we.frame_count - 1)) * dur;
        mrw_clip_to_palette(&skel, &walk, t, scratch, paletteA, jc);
        for (uint32_t b = 0; b < jc; ++b) mrw_baked_sample_bone(&bv, walk_ci, b, t, &paletteB[b * 12]);
        palette_diff(paletteA, paletteB, jc, &mb, &tmm);
        int pass = tmm < 1.0;   /* sub-mm */
        fprintf(stderr, "(1) exact-at-frame   TierA vs TierB : trans %.4f mm, basis %.2e  -> %s\n",
                tmm, mb, pass ? "PASS" : "FAIL");
        if (!pass) rc = 1;
    }

    /* (2) BETWEEN-FRAME error: same off-frame -> the chord-vs-arc gap (reported, not asserted) */
    {
        float t = (1.5f / (float)(we.frame_count - 1)) * dur;
        mrw_clip_to_palette(&skel, &walk, t, scratch, paletteA, jc);
        for (uint32_t b = 0; b < jc; ++b) mrw_baked_sample_bone(&bv, walk_ci, b, t, &paletteB[b * 12]);
        palette_diff(paletteA, paletteB, jc, &mb, &tmm);
        fprintf(stderr, "(2) between-frame    TierA vs TierB : trans %.4f mm  (chord-vs-arc; expected small, nonzero)\n", tmm);
    }

    /* (3) CROSS-FADE gap: CPU local-pose blend vs the baked component-space reference */
    {
        float t = 0.3f * dur, w = 0.5f;
        /* CPU: blend locals (marrow's blend primitive, in place), then compose */
        mrw_clip_sample_local(&walk, t, localsW, jc);
        mrw_clip_sample_local(&run,  t, localsR, jc);
        mrw_pose_blend(localsW, localsR, w, NULL, localsW, jc, jc);
        mrw_local_to_model(&skel, localsW, model12, jc);
        mrw_model_to_palette(&skel, model12, paletteA, jc);
        /* baked: component-space nlerp of the two baked clips, then compose (the shader's path) */
        for (uint32_t b = 0; b < jc; ++b) {
            mrw_xform xa, xb, xc;
            mrw_bake_sample_xform(texels, bv.frame_stride_texels, we.first_frame, we.frame_count,
                                  we.source_duration_s, (we.flags & MRW_BAKED_CLIP_LOOPING) != 0, b, t, &xa);
            mrw_bake_sample_xform(texels, bv.frame_stride_texels, re.first_frame, re.frame_count,
                                  re.source_duration_s, (re.flags & MRW_BAKED_CLIP_LOOPING) != 0, b, t, &xb);
            mrw_xform_nlerp(&xa, &xb, w, &xc);
            mrw_xform_to_affine(&xc, &paletteB[b * 12]);
        }
        palette_diff(paletteA, paletteB, jc, &mb, &tmm);
        fprintf(stderr, "(3) cross-fade gap   TierA-blend vs TierB-blend (w=0.5): trans %.4f mm  (measured bounded gap)\n", tmm);
    }

    mrw_authoring_free(localsW); mrw_authoring_free(localsR); mrw_authoring_free(model12);

    /* (4) BACKEND BENCHMARK + SIMD parity (shared helper, also used by --bench). */
    if (validate_microbench(&skel, &walk, dur, bench_n, stderr) != 0) rc = 1;

    mrw_authoring_free(scratch); mrw_authoring_free(paletteA); mrw_authoring_free(paletteB);

    /* (5) CPU-CROWD LAYOUT PARITY: the live CPU tier's AoS palette row layout vs the reference. */
    if (validate_cpu_crowd_layout(assets) != 0) rc = 1;

    /* (6) CONCURRENCY PARITY: the batch fanned across a thread pool == the serial batch, bitwise. */
    if (validate_jobs_parity(assets) != 0) rc = 1;

    /* (7) f16 PALETTE PARITY: decode the f16 palette and check it against the f32 palette. */
    if (validate_cpu_crowd_f16(assets) != 0) rc = 1;

    fprintf(stderr, "=== validation %s ===\n\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}

/* ------------------------------------------------------------------ shared CPU microbench */

int validate_microbench(const mrw_skeleton_view *skel, const mrw_clip_view *clip,
                        float dur, uint32_t N, FILE *out) {
    uint32_t jc = skel->joint_count;
    int rc = 0;
    mrw_mem_req sreq, preq;
    if (mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F32, &sreq, &preq) != MRW_OK) {
        fprintf(out, "(microbench) requirements query failed\n"); return 1;
    }
    void  *bscratch = mrw_authoring_alloc(sreq.size);
    float *pal_ref  = (float *)mrw_authoring_alloc(preq.size);
    float *pal_simd = (float *)mrw_authoring_alloc(preq.size);
    float *times    = (float *)mrw_authoring_alloc((size_t)N * sizeof(float));
    if (!bscratch || !pal_ref || !pal_simd || !times) {
        fprintf(out, "(microbench) scratch alloc failed\n");
        mrw_authoring_free(bscratch); mrw_authoring_free(pal_ref);
        mrw_authoring_free(pal_simd); mrw_authoring_free(times);
        return 1;
    }
    for (uint32_t i = 0; i < N; ++i) times[i] = fmodf((float)i * 0.0009f, dur);

    struct { const char *name; mrw_result (*make)(mrw_dispatch *); } bk[3] = {
        { "scalar", mrw_dispatch_scalar }, { "SSE2", mrw_dispatch_sse2 }, { "AVX2", mrw_dispatch_avx2 }
    };
    mrw_dispatch best; mrw_dispatch_detect(&best);
    const char *best_name = best.backend == MRW_BACKEND_AVX2 ? "AVX2" :
                            best.backend == MRW_BACKEND_SSE2 ? "SSE2" : "scalar";
    fprintf(out, "backend microbench: %u instances x %u bones (host best: %s)\n", N, jc, best_name);

    double scalar_ms = 0.0;
    for (int k = 0; k < 3; ++k) {
        mrw_dispatch disp;
        if (bk[k].make(&disp) != MRW_OK) { fprintf(out, "    %-6s: unsupported on this host\n", bk[k].name); continue; }
        float *o = (k == 0) ? pal_ref : pal_simd;
        mrw_batch_clip_to_palette(&disp, skel, clip, times, N, o, preq.size, bscratch, sreq.size); /* warm */
        const int R = 30;
        double t0 = prof_now_s();
        for (int r = 0; r < R; ++r)
            mrw_batch_clip_to_palette(&disp, skel, clip, times, N, o, preq.size, bscratch, sreq.size);
        double ms = (prof_now_s() - t0) / R * 1000.0;
        double ns_per = ms * 1e6 / ((double)N * jc);
        if (k == 0) scalar_ms = ms;
        double speedup = scalar_ms > 0.0 ? scalar_ms / ms : 1.0;
        double pmax_b = 0.0, pmax_t = 0.0;
        if (k > 0) palette_diff(pal_ref, pal_simd, N * jc, &pmax_b, &pmax_t);
        if (k == 0)
            fprintf(out, "    %-6s: %7.3f ms  (%.1f ns/inst-bone)\n", bk[k].name, ms, ns_per);
        else {
            int parity_ok = pmax_t < 1.0;   /* visual-only determinism: small FMA/reassoc gap */
            if (!parity_ok) rc = 1;
            fprintf(out, "    %-6s: %7.3f ms  (%.1f ns/inst-bone, %.2fx)  parity vs scalar: %.4f mm %s\n",
                    bk[k].name, ms, ns_per, speedup, pmax_t, parity_ok ? "OK" : "FAIL");
        }
    }
    mrw_authoring_free(bscratch); mrw_authoring_free(pal_ref); mrw_authoring_free(pal_simd); mrw_authoring_free(times);
    return rc;
}

/* ------------------------------------------------------------------ CPU-crowd layout parity */

int validate_cpu_crowd_layout(const ProcAssets *assets) {
    mrw_blob blob;
    if (mrw_blob_open(assets->blob, assets->blob_size, &blob) != MRW_OK) return 1;
    mrw_skeleton_view skel; mrw_blob_skeleton(&blob, &skel);
    mrw_clip_view walk; uint32_t walk_ci;
    if (!find_clip(assets, "walk", &blob, &walk, &walk_ci)) { fprintf(stderr, "(5) cpu-layout: no walk clip\n"); return 1; }

    uint32_t jc = skel.joint_count;
    const uint32_t N = 8;                    /* a few instances, off-frame phases */
    float dur = (walk.sample_count > 1 && walk.fps > 0.0f)
              ? (float)(walk.sample_count - 1) / walk.fps : 1.0f;

    mrw_mem_req sreq, preq;
    if (mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F32, &sreq, &preq) != MRW_OK) return 1;
    void  *bscratch = mrw_authoring_alloc(sreq.size);
    float *batch    = (float *)mrw_authoring_alloc(preq.size);                 /* CPU-tier output */
    float *oscratch = (float *)mrw_authoring_alloc((size_t)jc * 12 * sizeof(float));
    float *opal     = (float *)mrw_authoring_alloc((size_t)jc * 12 * sizeof(float)); /* per-instance reference */
    float *times    = (float *)mrw_authoring_alloc((size_t)N * sizeof(float));
    if (!bscratch || !batch || !oscratch || !opal || !times) {
        mrw_authoring_free(bscratch); mrw_authoring_free(batch); mrw_authoring_free(oscratch);
        mrw_authoring_free(opal); mrw_authoring_free(times); return 1;
    }
    for (uint32_t i = 0; i < N; ++i) times[i] = fmodf((float)i * 0.137f + 0.013f, dur);

    mrw_dispatch disp; mrw_dispatch_scalar(&disp);
    int rc = 0;
    if (mrw_batch_clip_to_palette(&disp, &skel, &walk, times, N, batch, preq.size, bscratch, sreq.size) != MRW_OK) {
        fprintf(stderr, "(5) cpu-layout: batch failed\n"); rc = 1;
    } else {
        const DemoVertex *V = assets->verts;
        double max_mm = 0.0;
        for (uint32_t i = 0; i < N; ++i) {
            mrw_clip_to_palette(&skel, &walk, times[i], oscratch, opal, jc);   /* per-instance reference */
            for (uint32_t v = 0; v < assets->vert_count; ++v) {
                float ac[3] = { 0, 0, 0 }, ao[3] = { 0, 0, 0 };
                for (int k = 0; k < 4; ++k) {
                    float wt = V[v].weights[k];
                    if (wt == 0.0f) continue;
                    uint32_t bone = V[v].bones[k];
                    /* index `batch` exactly as skin_tierA.vert does: rows at (i*jc + bone)*12 */
                    const float *Mc = &batch[((size_t)i * jc + bone) * 12];
                    const float *Mo = &opal[(size_t)bone * 12];
                    const float *p = V[v].pos;
                    for (int r = 0; r < 3; ++r) {
                        ac[r] += wt * (Mc[r*4]*p[0] + Mc[r*4+1]*p[1] + Mc[r*4+2]*p[2] + Mc[r*4+3]);
                        ao[r] += wt * (Mo[r*4]*p[0] + Mo[r*4+1]*p[1] + Mo[r*4+2]*p[2] + Mo[r*4+3]);
                    }
                }
                double dx = ac[0]-ao[0], dy = ac[1]-ao[1], dz = ac[2]-ao[2];
                double mm = sqrt(dx*dx + dy*dy + dz*dz) * 1000.0;
                if (mm > max_mm) max_mm = mm;
            }
        }
        /* scalar batch == per-instance loop bit-for-bit, so any nonzero gap is a layout bug. */
        int pass = max_mm < 1e-3;
        fprintf(stderr, "(5) CPU-crowd layout parity (%u inst x %u verts) : max %.6f mm  -> %s\n",
                N, assets->vert_count, max_mm, pass ? "PASS" : "FAIL");
        if (!pass) rc = 1;
    }
    mrw_authoring_free(bscratch); mrw_authoring_free(batch); mrw_authoring_free(oscratch);
    mrw_authoring_free(opal); mrw_authoring_free(times);
    return rc;
}

/* ------------------------------------------------------------------ f16 palette parity */

int validate_cpu_crowd_f16(const ProcAssets *assets) {
    mrw_blob blob;
    if (mrw_blob_open(assets->blob, assets->blob_size, &blob) != MRW_OK) return 1;
    mrw_skeleton_view skel; mrw_blob_skeleton(&blob, &skel);
    mrw_clip_view walk; uint32_t walk_ci;
    if (!find_clip(assets, "walk", &blob, &walk, &walk_ci)) { fprintf(stderr, "(7) f16-parity: no walk clip\n"); return 1; }

    uint32_t jc = skel.joint_count;
    const uint32_t N = 64;                    /* a spread of off-frame phases across many joints */
    float dur = (walk.sample_count > 1 && walk.fps > 0.0f)
              ? (float)(walk.sample_count - 1) / walk.fps : 1.0f;

    mrw_mem_req sreq, preq, preq16;
    if (mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F32, &sreq, &preq) != MRW_OK ||
        mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F16, NULL, &preq16) != MRW_OK) return 1;
    void     *bscratch = mrw_authoring_alloc(sreq.size);
    float    *pal32    = (float *)mrw_authoring_alloc(preq.size);
    uint16_t *pal16    = (uint16_t *)mrw_authoring_alloc(preq16.size);
    uint16_t *pal16b   = (uint16_t *)mrw_authoring_alloc(preq16.size);  /* generated in two sub-ranges */
    float    *pal16dec = (float *)mrw_authoring_alloc(preq.size);    /* f16 widened back to f32 */
    float    *times    = (float *)mrw_authoring_alloc((size_t)N * sizeof(float));
    if (!bscratch || !pal32 || !pal16 || !pal16b || !pal16dec || !times) {
        mrw_authoring_free(bscratch); mrw_authoring_free(pal32); mrw_authoring_free(pal16);
        mrw_authoring_free(pal16b); mrw_authoring_free(pal16dec); mrw_authoring_free(times); return 1;
    }
    for (uint32_t i = 0; i < N; ++i) times[i] = fmodf((float)i * 0.0137f + 0.005f, dur);

    /* scalar always (the f16 store is bit-exactly mrw_f32_to_f16 of the f32 reference); the host-best
     * backend too - its f16 output is round(its own f32), so the decode gap is f16 rounding for any
     * backend. Tolerance: ½ binary16 quantization (the f32<->f16 gap, NOT a SIMD reassoc gap, since
     * we compare each backend's f16 against the SAME backend's f32). */
    struct { const char *name; mrw_result (*make)(mrw_dispatch *); } bk[2];
    int nbk = 0;
    bk[nbk].name = "scalar"; bk[nbk].make = mrw_dispatch_scalar; nbk++;
    mrw_dispatch best; mrw_dispatch_detect(&best);
    if (best.backend != MRW_BACKEND_SCALAR) {
        bk[nbk].name = best.backend == MRW_BACKEND_AVX2 ? "AVX2" : "SSE2";
        bk[nbk].make = best.backend == MRW_BACKEND_AVX2 ? mrw_dispatch_avx2 : mrw_dispatch_sse2;
        nbk++;
    }

    const size_t ncomp = (size_t)N * jc * 12u;
    /* ≤ ½ ULP/component → trans ≤ ~1 cm p100 and basis a few ×1e-4. */
    const double TRANS_MM_MAX = 10.0, BASIS_MAX = 4e-3;
    int rc = 0;
    for (int k = 0; k < nbk; ++k) {
        mrw_dispatch disp;
        if (bk[k].make(&disp) != MRW_OK) continue;
        if (mrw_batch_clip_to_palette(&disp, &skel, &walk, times, N, pal32, preq.size, bscratch, sreq.size) != MRW_OK ||
            mrw_batch_clip_to_palette_f16(&disp, &skel, &walk, times, N, pal16, preq16.size, bscratch, sreq.size) != MRW_OK) {
            fprintf(stderr, "(7) f16-parity [%s]: batch failed\n", bk[k].name); rc = 1; continue;
        }
        for (size_t i = 0; i < ncomp; ++i) pal16dec[i] = mrw_half_to_float(pal16[i]);
        double mb, tmm;
        palette_diff(pal32, pal16dec, N * jc, &mb, &tmm);
        int pass = tmm < TRANS_MM_MAX && mb < BASIS_MAX;

        /* Partition parity (the live crowd_cpu gen pattern): the SAME f16 output written as two
         * disjoint sub-ranges [0,s)+[s,N) into one buffer in place must be bit-identical to the
         * whole-batch result - each instance's entry is a pure function of its own time. Exercises
         * the demo's interior-offset f16 write (8-byte aligned on odd jc - the relaxed alignment). */
        const uint32_t s = 13;   /* odd, non-MRW_LANES-aligned split */
        int part_ok = 1;
        if (mrw_batch_clip_to_palette_f16(&disp, &skel, &walk, times, s, pal16b, preq16.size, bscratch, sreq.size) != MRW_OK ||
            mrw_batch_clip_to_palette_f16(&disp, &skel, &walk, times + s, N - s,
                pal16b + (size_t)s * jc * 12u, preq16.size - (size_t)s * jc * 12u * sizeof(uint16_t),
                bscratch, sreq.size) != MRW_OK ||
            memcmp(pal16, pal16b, ncomp * sizeof(uint16_t)) != 0) {
            part_ok = 0; pass = 0;
        }
        fprintf(stderr, "(7) f16 palette parity [%-6s] %u inst x %u bones : trans %.4f mm, basis %.2e, partition %s -> %s\n",
                bk[k].name, N, jc, tmm, mb, part_ok ? "ok" : "MISMATCH", pass ? "PASS" : "FAIL");
        if (!pass) rc = 1;
    }

    mrw_authoring_free(bscratch); mrw_authoring_free(pal32); mrw_authoring_free(pal16);
    mrw_authoring_free(pal16b); mrw_authoring_free(pal16dec); mrw_authoring_free(times);
    return rc;
}

/* ------------------------------------------------------------------ concurrency parity (jobified) */

/* One job lane: run mrw_batch_clip_to_palette over its instance sub-range, into a disjoint output
 * slice, with its OWN scratch slice. The shared inputs (skel/clip views, dispatch, times) are all
 * read-only - the exact safe-fan-out contract from jobs.h. */
typedef struct {
    const mrw_skeleton_view *skel; const mrw_clip_view *clip; const float *times;
    const mrw_dispatch *disp; float *out; size_t out_bytes;
    void *scratch; size_t unit; uint32_t jc;
    mrw_result err[JOBS_MAX_WORKERS];
} ParityCtx;

static void parity_range(void *vctx, uint32_t worker, uint32_t begin, uint32_t end) {
    ParityCtx *p = (ParityCtx *)vctx;
    if (end <= begin) return;
    size_t off = (size_t)begin * p->jc * 12u;
    void *sc = (char *)p->scratch + (size_t)worker * p->unit;
    mrw_result r = mrw_batch_clip_to_palette(p->disp, p->skel, p->clip,
            &p->times[begin], end - begin,
            p->out + off, p->out_bytes - off * sizeof(float),
            sc, p->unit);
    if (r != MRW_OK && p->err[worker] == MRW_OK) p->err[worker] = r;
}

int validate_jobs_parity(const ProcAssets *assets) {
    mrw_blob blob;
    if (mrw_blob_open(assets->blob, assets->blob_size, &blob) != MRW_OK) return 1;
    mrw_skeleton_view skel; mrw_blob_skeleton(&blob, &skel);
    mrw_clip_view walk; uint32_t walk_ci;
    if (!find_clip(assets, "walk", &blob, &walk, &walk_ci)) { fprintf(stderr, "(6) jobs-parity: no walk clip\n"); return 1; }

    uint32_t jc = skel.joint_count;
    /* N is PRIME (not a multiple of MRW_LANES=8), so the per-lane chunk boundaries land OFF the
     * 8-wide tile grid on every host: each lane's batch call gets a partial final tile and instances
     * sit at different lane positions than in the serial run - exactly the partition/tail cases the
     * bit-for-bit claim must survive. */
    const uint32_t N = 16381u;
    float dur = (walk.sample_count > 1 && walk.fps > 0.0f)
              ? (float)(walk.sample_count - 1) / walk.fps : 1.0f;

    /* Use the host pool (realistic speedup), but FORCE ≥2 lanes so the gate always exercises real
     * concurrent calls - even on a single-core host or CI runner. */
    Jobs *pool = jobs_create(0);
    uint32_t lanes = jobs_worker_count(pool);
    if (lanes < 2) { jobs_destroy(pool); pool = jobs_create(2); lanes = jobs_worker_count(pool); }
    if (lanes < 2) {   /* pathological: a background thread refused to start - can't test concurrency */
        fprintf(stderr, "(6) jobs parity: SKIPPED (could not create a multi-lane pool)\n");
        jobs_destroy(pool); return 0;
    }

    mrw_mem_req sreq, preq;
    if (mrw_batch_clip_to_palette_requirements(jc, N, MRW_PALETTE_F32, &sreq, &preq) != MRW_OK) { jobs_destroy(pool); return 1; }

    void  *serial_scratch = mrw_authoring_alloc(sreq.size);
    void  *pool_scratch   = mrw_authoring_alloc(sreq.size * lanes);   /* one slice per lane */
    float *pal_ref = (float *)mrw_authoring_alloc(preq.size);
    float *pal_mt  = (float *)mrw_authoring_alloc(preq.size);
    float *times   = (float *)mrw_authoring_alloc((size_t)N * sizeof(float));
    if (!serial_scratch || !pool_scratch || !pal_ref || !pal_mt || !times) {
        mrw_authoring_free(serial_scratch); mrw_authoring_free(pool_scratch);
        mrw_authoring_free(pal_ref); mrw_authoring_free(pal_mt); mrw_authoring_free(times);
        jobs_destroy(pool); return 1;
    }
    for (uint32_t i = 0; i < N; ++i) times[i] = fmodf((float)i * 0.0011f + 0.003f, dur);

    /* scalar always; the host-best backend too (proves SIMD output is partition-independent). */
    struct { const char *name; mrw_result (*make)(mrw_dispatch *); } bk[2];
    int nbk = 0;
    bk[nbk].name = "scalar"; bk[nbk].make = mrw_dispatch_scalar; nbk++;
    mrw_dispatch best; mrw_dispatch_detect(&best);
    if (best.backend != MRW_BACKEND_SCALAR) {
        bk[nbk].name = best.backend == MRW_BACKEND_AVX2 ? "AVX2" : "SSE2";
        bk[nbk].make = best.backend == MRW_BACKEND_AVX2 ? mrw_dispatch_avx2 : mrw_dispatch_sse2;
        nbk++;
    }

    const int R = 20;   /* timing reps (informational; the PASS/FAIL gate is parity, not speed) */
    int rc = 0;
    for (int k = 0; k < nbk; ++k) {
        mrw_dispatch disp;
        if (bk[k].make(&disp) != MRW_OK) continue;

        ParityCtx p; memset(&p, 0, sizeof p);
        p.skel = &skel; p.clip = &walk; p.times = times; p.disp = &disp;
        p.out = pal_mt; p.out_bytes = preq.size; p.scratch = pool_scratch; p.unit = sreq.size; p.jc = jc;

        /* serial reference: one call over all N instances (warm, then time) */
        if (mrw_batch_clip_to_palette(&disp, &skel, &walk, times, N, pal_ref, preq.size,
                                      serial_scratch, sreq.size) != MRW_OK) {
            fprintf(stderr, "(6) jobs-parity [%s]: serial batch failed\n", bk[k].name); rc = 1; continue;
        }
        double t0 = prof_now_s();
        for (int r = 0; r < R; ++r)
            mrw_batch_clip_to_palette(&disp, &skel, &walk, times, N, pal_ref, preq.size,
                                      serial_scratch, sreq.size);
        double serial_ms = (prof_now_s() - t0) / R * 1000.0;

        /* threaded: the SAME work fanned across the pool - disjoint output, per-lane scratch */
        if (pool) jobs_parallel_for(pool, N, parity_range, &p);   /* warm */
        else      parity_range(&p, 0, 0, N);
        t0 = prof_now_s();
        for (int r = 0; r < R; ++r) {
            if (pool) jobs_parallel_for(pool, N, parity_range, &p);
            else      parity_range(&p, 0, 0, N);
        }
        double thr_ms = (prof_now_s() - t0) / R * 1000.0;

        int lane_err = 0;
        for (uint32_t w = 0; w < lanes; ++w) if (p.err[w] != MRW_OK) lane_err = 1;

        /* Bit-for-bit: each instance's result is a pure function of its own time + the shared
         * read-only skeleton/clip, so partitioning must not change a single bit (any difference is a
         * data race or an OOB write across lane scratch/output). On a lane error a slice may be
         * unwritten, so only scan/compare pal_mt when every lane succeeded. */
        size_t nfloat = preq.size / sizeof(float);
        uint32_t mism = 0;
        if (!lane_err)
            for (size_t i = 0; i < nfloat; ++i) if (pal_ref[i] != pal_mt[i]) mism++;
        int pass = !lane_err && memcmp(pal_ref, pal_mt, preq.size) == 0;
        double speedup = thr_ms > 0.0 ? serial_ms / thr_ms : 0.0;
        fprintf(stderr, "(6) jobs parity [%-6s] %u lanes x %u inst : %s%s | serial %.3f ms  jobs %.3f ms  (%.2fx)\n",
                bk[k].name, lanes, N, pass ? "PASS" : "FAIL",
                lane_err ? " (lane error)" : (mism ? " (bits differ)" : ""),
                serial_ms, thr_ms, speedup);
        if (!pass) rc = 1;
    }

    mrw_authoring_free(serial_scratch); mrw_authoring_free(pool_scratch);
    mrw_authoring_free(pal_ref); mrw_authoring_free(pal_mt); mrw_authoring_free(times);
    jobs_destroy(pool);
    return rc;
}

/* ------------------------------------------------------------------ GPU readback */

typedef struct {
    uint32_t firstFrame, frameCount, looping, texelsPerBone;
    float    t, dur;
    uint32_t vertCount, _pad;
} CompPush;
_Static_assert(sizeof(CompPush) == 32, "CompPush must be 32 bytes");
_Static_assert(offsetof(CompPush, firstFrame) ==  0, "CompPush.firstFrame");
_Static_assert(offsetof(CompPush, t)          == 16, "CompPush.t");
_Static_assert(offsetof(CompPush, dur)        == 20, "CompPush.dur");
_Static_assert(offsetof(CompPush, vertCount)  == 24, "CompPush.vertCount");

int validate_gpu(VkCtx *ctx, const ProcAssets *assets) {
    mrw_blob blob;
    if (mrw_blob_open(assets->blob, assets->blob_size, &blob) != MRW_OK) return 1;
    mrw_baked_view bv; if (!find_baked(&blob, &bv)) return 1;
    mrw_clip_view walk; uint32_t walk_ci;
    if (!find_clip(assets, "walk", &blob, &walk, &walk_ci)) return 1;
    mrw_baked_clip we; mrw_baked_clip_entry(&bv, walk_ci, &we);

    uint32_t vc = assets->vert_count;
    float t = 0.37f * we.source_duration_s;   /* off-frame: exercises temporal interp */

    /* palette texture + sampler */
    VkImage pal_img; VkDeviceMemory pal_mem; VkImageView pal_view;
    if (!vkc_create_texture_rgba16f(ctx, bv.frame_stride_texels, bv.total_frames,
            bv.base + bv.texels_off, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            &pal_img, &pal_mem, &pal_view)) return 1;
    VkSampler sampler;
    { VkSamplerCreateInfo si = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
      si.magFilter = si.minFilter = VK_FILTER_NEAREST; si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      if (vkCreateSampler(ctx->device, &si, NULL, &sampler) != VK_SUCCESS) return 1; }

    /* vertex SSBO (in) + position SSBO (out, host-visible) */
    VkBuffer vbuf, obuf; VkDeviceMemory vmem, omem; void *vp, *op;
    if (!vkc_create_host_buffer(ctx, (VkDeviceSize)vc * sizeof(DemoVertex), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vbuf, &vmem, &vp)) return 1;
    memcpy(vp, assets->verts, (size_t)vc * sizeof(DemoVertex));
    if (!vkc_create_host_buffer(ctx, (VkDeviceSize)vc * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &obuf, &omem, &op)) return 1;

    /* descriptor set: sampler + 2 SSBOs */
    VkDescriptorSetLayoutBinding binds[3] = { 0 };
    binds[0].binding = 0; binds[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; binds[0].descriptorCount = 1; binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1; binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; binds[1].descriptorCount = 1; binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[2].binding = 2; binds[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; binds[2].descriptorCount = 1; binds[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo li = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    li.bindingCount = 3; li.pBindings = binds;
    VkDescriptorSetLayout set_layout;
    if (vkCreateDescriptorSetLayout(ctx->device, &li, NULL, &set_layout) != VK_SUCCESS) return 1;

    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 } };
    VkDescriptorPoolCreateInfo pi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.maxSets = 1; pi.poolSizeCount = 2; pi.pPoolSizes = ps;
    VkDescriptorPool pool;
    if (vkCreateDescriptorPool(ctx->device, &pi, NULL, &pool) != VK_SUCCESS) return 1;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &set_layout;
    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(ctx->device, &ai, &set) != VK_SUCCESS) return 1;

    VkDescriptorImageInfo ii = { sampler, pal_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorBufferInfo vi = { vbuf, 0, VK_WHOLE_SIZE }, oi = { obuf, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w[3] = { 0 };
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[0].dstSet = set; w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &ii;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[1].dstSet = set; w[1].dstBinding = 1; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &vi;
    w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[2].dstSet = set; w[2].dstBinding = 2; w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[2].pBufferInfo = &oi;
    vkUpdateDescriptorSets(ctx->device, 3, w, 0, NULL);

    /* compute shader object + pipeline layout */
    VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CompPush) };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1; pli.pSetLayouts = &set_layout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pr;
    VkPipelineLayout play;
    if (vkCreatePipelineLayout(ctx->device, &pli, NULL, &play) != VK_SUCCESS) return 1;

    VkShaderCreateInfoEXT sci = { VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT };
    sci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    sci.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
    sci.pCode = validate_skin_comp_spv; sci.codeSize = sizeof validate_skin_comp_spv;
    sci.pName = "main";
    sci.setLayoutCount = 1; sci.pSetLayouts = &set_layout;
    sci.pushConstantRangeCount = 1; sci.pPushConstantRanges = &pr;
    VkShaderEXT comp;
    if (vkCreateShadersEXT(ctx->device, 1, &sci, NULL, &comp) != VK_SUCCESS) { fprintf(stderr, "[validate-gpu] compute shader create failed\n"); return 1; }

    /* dispatch */
    VkCommandBuffer cmd = vkc_single_time_begin(ctx);
    VkShaderStageFlagBits cs = VK_SHADER_STAGE_COMPUTE_BIT;
    vkCmdBindShadersEXT(cmd, 1, &cs, &comp);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, play, 0, 1, &set, 0, NULL);
    CompPush cp = { we.first_frame, we.frame_count, (we.flags & MRW_BAKED_CLIP_LOOPING) ? 1u : 0u, 2u, t, we.source_duration_s, vc, 0u };
    vkCmdPushConstants(cmd, play, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof cp, &cp);
    vkCmdDispatch(cmd, (vc + 63) / 64, 1, 1);
    VkMemoryBarrier2 mbar = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    mbar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; mbar.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mbar.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT; mbar.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO }; di.memoryBarrierCount = 1; di.pMemoryBarriers = &mbar;
    vkCmdPipelineBarrier2(cmd, &di);
    vkc_single_time_end(ctx, cmd);

    /* CPU reference: Σ wᵢ · (Mᵢ · [p,1]) with Mᵢ = mrw_baked_sample_bone */
    const float *gpu = (const float *)op;
    const DemoVertex *V = assets->verts;
    double max_mm = 0.0;
    for (uint32_t i = 0; i < vc; ++i) {
        float acc[3] = { 0, 0, 0 };
        for (int k = 0; k < 4; ++k) {
            float wt = V[i].weights[k];
            if (wt == 0.0f) continue;
            float M[12]; mrw_baked_sample_bone(&bv, walk_ci, V[i].bones[k], t, M);
            const float *p = V[i].pos;
            acc[0] += wt * (M[0]*p[0] + M[1]*p[1] + M[2]*p[2]  + M[3]);
            acc[1] += wt * (M[4]*p[0] + M[5]*p[1] + M[6]*p[2]  + M[7]);
            acc[2] += wt * (M[8]*p[0] + M[9]*p[1] + M[10]*p[2] + M[11]);
        }
        double dx = acc[0] - gpu[i*4+0], dy = acc[1] - gpu[i*4+1], dz = acc[2] - gpu[i*4+2];
        double mm = sqrt(dx*dx + dy*dy + dz*dz) * 1000.0;
        if (mm > max_mm) max_mm = mm;
    }
    int pass = max_mm < 1.0;
    fprintf(stderr, "(5) GPU skin vs CPU oracle (%u verts, off-frame) : max %.4f mm  -> %s\n",
            vc, max_mm, pass ? "PASS" : "FAIL");

    vkDestroyShaderEXT(ctx->device, comp, NULL);
    vkDestroyPipelineLayout(ctx->device, play, NULL);
    vkDestroyDescriptorPool(ctx->device, pool, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, set_layout, NULL);
    vkDestroySampler(ctx->device, sampler, NULL);
    vkDestroyImageView(ctx->device, pal_view, NULL);
    vkDestroyImage(ctx->device, pal_img, NULL);
    vkFreeMemory(ctx->device, pal_mem, NULL);
    vkc_destroy_buffer(ctx, vbuf, vmem);
    vkc_destroy_buffer(ctx, obuf, omem);
    return pass ? 0 : 1;
}

/* ------------------------------------------------------------------ f16 GPU readback */

typedef struct { uint32_t vertCount; } CompPushF16;          /* texture-skin pass */
_Static_assert(sizeof(CompPushF16) == 4, "CompPushF16 must be 4 bytes");
typedef struct { VkDeviceAddress palette, outp; uint32_t jointCount, _pad; } DecPushF16;  /* SSBO decode pass */
_Static_assert(sizeof(DecPushF16) == 24, "DecPushF16 must be 24 bytes");

int validate_gpu_f16(VkCtx *ctx, const ProcAssets *assets) {
    mrw_blob blob;
    if (mrw_blob_open(assets->blob, assets->blob_size, &blob) != MRW_OK) return 1;
    mrw_skeleton_view skel; mrw_blob_skeleton(&blob, &skel);
    mrw_clip_view walk; uint32_t walk_ci;
    if (!find_clip(assets, "walk", &blob, &walk, &walk_ci)) { fprintf(stderr, "[validate-gpu-f16] no walk clip\n"); return 1; }

    uint32_t jc = skel.joint_count, vc = assets->vert_count;
    float dur = (walk.sample_count > 1 && walk.fps > 0.0f)
              ? (float)(walk.sample_count - 1) / walk.fps : 1.0f;
    float t = 0.37f * dur;   /* off-frame: a general (non-baked-frame) pose */

    /* all Vulkan handles NULL-init so the single cleanup path can unconditionally destroy them. */
    VkImage pal_img = VK_NULL_HANDLE; VkDeviceMemory pal_mem = VK_NULL_HANDLE; VkImageView pal_view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkBuffer vbuf = VK_NULL_HANDLE, obuf = VK_NULL_HANDLE; VkDeviceMemory vmem = VK_NULL_HANDLE, omem = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE; VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout play = VK_NULL_HANDLE; VkShaderEXT comp = VK_NULL_HANDLE;
    VkBuffer palbuf = VK_NULL_HANDLE, decbuf = VK_NULL_HANDLE; VkDeviceMemory palmem = VK_NULL_HANDLE, decmem = VK_NULL_HANDLE;
    VkPipelineLayout slay = VK_NULL_HANDLE; VkShaderEXT scomp = VK_NULL_HANDLE;
    void *vp = NULL, *op = NULL, *palp = NULL, *decp = NULL;
    int rc = 1;   /* default failure until both passes pass */

    /* one instance's CPU f16 full-matrix palette (scalar = bit-exact mrw_f32_to_f16 of the reference) */
    mrw_mem_req sreq, preq16;
    if (mrw_batch_clip_to_palette_requirements(jc, 1, MRW_PALETTE_F16, &sreq, &preq16) != MRW_OK) return 1;
    void     *bscratch = mrw_authoring_alloc(sreq.size);
    uint16_t *pal16    = (uint16_t *)mrw_authoring_alloc(preq16.size);
    float    *oscratch = (float *)mrw_authoring_alloc((size_t)jc * 12 * sizeof(float));
    float    *opal     = (float *)mrw_authoring_alloc((size_t)jc * 12 * sizeof(float));
    if (!bscratch || !pal16 || !oscratch || !opal) goto cleanup;
    mrw_dispatch disp; mrw_dispatch_scalar(&disp);
    if (mrw_batch_clip_to_palette_f16(&disp, &skel, &walk, &t, 1, pal16, preq16.size, bscratch, sreq.size) != MRW_OK) {
        fprintf(stderr, "[validate-gpu-f16] f16 batch failed\n"); goto cleanup;
    }
    mrw_clip_to_palette(&skel, &walk, t, oscratch, opal, jc);   /* f32 reference palette, shared by both passes */

    /* ---- pass (6): RGBA16F texture / texelFetch (the 3-texel layout + RGBA16F decode + skin) ---- */
    int tex_pass = 0;
    if (!vkc_create_texture_rgba16f(ctx, jc * 3u, 1u, pal16,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, &pal_img, &pal_mem, &pal_view)) goto cleanup;
    { VkSamplerCreateInfo si = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
      si.magFilter = si.minFilter = VK_FILTER_NEAREST; si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      if (vkCreateSampler(ctx->device, &si, NULL, &sampler) != VK_SUCCESS) goto cleanup; }

    if (!vkc_create_host_buffer(ctx, (VkDeviceSize)vc * sizeof(DemoVertex), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vbuf, &vmem, &vp)) goto cleanup;
    memcpy(vp, assets->verts, (size_t)vc * sizeof(DemoVertex));
    if (!vkc_create_host_buffer(ctx, (VkDeviceSize)vc * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &obuf, &omem, &op)) goto cleanup;

    VkDescriptorSetLayoutBinding binds[3] = { 0 };
    binds[0].binding = 0; binds[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; binds[0].descriptorCount = 1; binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1; binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; binds[1].descriptorCount = 1; binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[2].binding = 2; binds[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; binds[2].descriptorCount = 1; binds[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo li = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    li.bindingCount = 3; li.pBindings = binds;
    if (vkCreateDescriptorSetLayout(ctx->device, &li, NULL, &set_layout) != VK_SUCCESS) goto cleanup;

    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 } };
    VkDescriptorPoolCreateInfo pi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.maxSets = 1; pi.poolSizeCount = 2; pi.pPoolSizes = ps;
    if (vkCreateDescriptorPool(ctx->device, &pi, NULL, &pool) != VK_SUCCESS) goto cleanup;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &set_layout;
    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(ctx->device, &ai, &set) != VK_SUCCESS) goto cleanup;

    VkDescriptorImageInfo ii = { sampler, pal_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorBufferInfo vi = { vbuf, 0, VK_WHOLE_SIZE }, oi = { obuf, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w[3] = { 0 };
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[0].dstSet = set; w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &ii;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[1].dstSet = set; w[1].dstBinding = 1; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &vi;
    w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[2].dstSet = set; w[2].dstBinding = 2; w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[2].pBufferInfo = &oi;
    vkUpdateDescriptorSets(ctx->device, 3, w, 0, NULL);

    VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CompPushF16) };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1; pli.pSetLayouts = &set_layout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(ctx->device, &pli, NULL, &play) != VK_SUCCESS) goto cleanup;

    VkShaderCreateInfoEXT sci = { VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT };
    sci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    sci.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
    sci.pCode = validate_skin_f16_comp_spv; sci.codeSize = sizeof validate_skin_f16_comp_spv;
    sci.pName = "main";
    sci.setLayoutCount = 1; sci.pSetLayouts = &set_layout;
    sci.pushConstantRangeCount = 1; sci.pPushConstantRanges = &pr;
    if (vkCreateShadersEXT(ctx->device, 1, &sci, NULL, &comp) != VK_SUCCESS) { fprintf(stderr, "[validate-gpu-f16] texture compute shader create failed\n"); goto cleanup; }

    { VkCommandBuffer cmd = vkc_single_time_begin(ctx);
      VkShaderStageFlagBits cs = VK_SHADER_STAGE_COMPUTE_BIT;
      vkCmdBindShadersEXT(cmd, 1, &cs, &comp);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, play, 0, 1, &set, 0, NULL);
      CompPushF16 cp = { vc };
      vkCmdPushConstants(cmd, play, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof cp, &cp);
      vkCmdDispatch(cmd, (vc + 63) / 64, 1, 1);
      VkMemoryBarrier2 mbar = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
      mbar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; mbar.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
      mbar.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT; mbar.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
      VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO }; di.memoryBarrierCount = 1; di.pMemoryBarriers = &mbar;
      vkCmdPipelineBarrier2(cmd, &di);
      vkc_single_time_end(ctx, cmd); }

    /* CPU reference: Σ wᵢ · (Mᵢ · [p,1]) with the f32 per-bone palette - the gap is pure f16 quantization */
    { const float *gpu = (const float *)op;
      const DemoVertex *V = assets->verts;
      double max_mm = 0.0;
      for (uint32_t i = 0; i < vc; ++i) {
          float acc[3] = { 0, 0, 0 };
          for (int k = 0; k < 4; ++k) {
              float wt = V[i].weights[k];
              if (wt == 0.0f) continue;
              const float *M = &opal[(size_t)V[i].bones[k] * 12];
              const float *p = V[i].pos;
              acc[0] += wt * (M[0]*p[0] + M[1]*p[1] + M[2]*p[2]  + M[3]);
              acc[1] += wt * (M[4]*p[0] + M[5]*p[1] + M[6]*p[2]  + M[7]);
              acc[2] += wt * (M[8]*p[0] + M[9]*p[1] + M[10]*p[2] + M[11]);
          }
          double dx = acc[0] - gpu[i*4+0], dy = acc[1] - gpu[i*4+1], dz = acc[2] - gpu[i*4+2];
          double mm = sqrt(dx*dx + dy*dy + dz*dz) * 1000.0;
          if (mm > max_mm) max_mm = mm;
      }
      tex_pass = max_mm < 10.0;   /* f16 quantization at character-local extent */
      fprintf(stderr, "(6) GPU f16 texture-skin (RGBA16F texelFetch) vs CPU f32 oracle (%u verts) : max %.4f mm  -> %s\n",
              vc, max_mm, tex_pass ? "PASS" : "FAIL"); }

    /* ---- pass (6b): the LIVE SSBO fetch - packed uvec2 + unpackHalf2x16 (skin_tierA_f16.vert's decode) ---- */
    int ssbo_pass = 0;
    VkBufferUsageFlags bda = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    size_t dec_bytes = (size_t)jc * 12u * sizeof(float);
    if (!vkc_create_host_buffer(ctx, (VkDeviceSize)preq16.size, bda, &palbuf, &palmem, &palp)) goto cleanup;
    memcpy(palp, pal16, (size_t)jc * 12u * sizeof(uint16_t));
    if (!vkc_create_host_buffer(ctx, (VkDeviceSize)dec_bytes, bda, &decbuf, &decmem, &decp)) goto cleanup;

    VkPushConstantRange spr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DecPushF16) };
    VkPipelineLayoutCreateInfo spli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    spli.pushConstantRangeCount = 1; spli.pPushConstantRanges = &spr;
    if (vkCreatePipelineLayout(ctx->device, &spli, NULL, &slay) != VK_SUCCESS) goto cleanup;

    VkShaderCreateInfoEXT ssci = { VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT };
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
    ssci.pCode = validate_skin_f16_ssbo_comp_spv; ssci.codeSize = sizeof validate_skin_f16_ssbo_comp_spv;
    ssci.pName = "main";
    ssci.pushConstantRangeCount = 1; ssci.pPushConstantRanges = &spr;
    if (vkCreateShadersEXT(ctx->device, 1, &ssci, NULL, &scomp) != VK_SUCCESS) { fprintf(stderr, "[validate-gpu-f16] ssbo compute shader create failed\n"); goto cleanup; }

    { VkCommandBuffer cmd = vkc_single_time_begin(ctx);
      VkShaderStageFlagBits cs = VK_SHADER_STAGE_COMPUTE_BIT;
      vkCmdBindShadersEXT(cmd, 1, &cs, &scomp);
      DecPushF16 dp = { vkc_buffer_address(ctx, palbuf), vkc_buffer_address(ctx, decbuf), jc, 0u };
      vkCmdPushConstants(cmd, slay, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof dp, &dp);
      vkCmdDispatch(cmd, (jc + 63) / 64, 1, 1);
      VkMemoryBarrier2 mbar = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
      mbar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; mbar.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
      mbar.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT; mbar.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
      VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO }; di.memoryBarrierCount = 1; di.pMemoryBarriers = &mbar;
      vkCmdPipelineBarrier2(cmd, &di);
      vkc_single_time_end(ctx, cmd); }

    /* GPU unpackHalf2x16 must reproduce mrw_half_to_float of the SAME bytes exactly (binary16→f32 is a
     * lossless widening); a wrong half lane / uvec2 stride / endianness would show here. (A tiny
     * tolerance absorbs an f16-denormal flush should any GPU do it.) */
    { const float *gpu = (const float *)decp;
      double max_abs = 0.0;
      for (size_t k = 0; k < (size_t)jc * 12u; ++k) {
          double d = fabs((double)gpu[k] - (double)mrw_half_to_float(pal16[k]));
          if (d > max_abs) max_abs = d;
      }
      ssbo_pass = max_abs < 1e-4;
      fprintf(stderr, "(6b) GPU f16 SSBO decode (unpackHalf2x16) vs CPU widen (%u joints) : max %.3e  -> %s\n",
              jc, max_abs, ssbo_pass ? "PASS" : "FAIL"); }

    rc = (tex_pass && ssbo_pass) ? 0 : 1;

cleanup:
    vkDestroyShaderEXT(ctx->device, scomp, NULL);
    vkDestroyPipelineLayout(ctx->device, slay, NULL);
    vkc_destroy_buffer(ctx, decbuf, decmem);
    vkc_destroy_buffer(ctx, palbuf, palmem);
    vkDestroyShaderEXT(ctx->device, comp, NULL);
    vkDestroyPipelineLayout(ctx->device, play, NULL);
    vkDestroyDescriptorPool(ctx->device, pool, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, set_layout, NULL);
    vkDestroySampler(ctx->device, sampler, NULL);
    vkDestroyImageView(ctx->device, pal_view, NULL);
    vkDestroyImage(ctx->device, pal_img, NULL);
    vkFreeMemory(ctx->device, pal_mem, NULL);
    vkc_destroy_buffer(ctx, vbuf, vmem);
    vkc_destroy_buffer(ctx, obuf, omem);
    mrw_authoring_free(bscratch); mrw_authoring_free(pal16);
    mrw_authoring_free(oscratch); mrw_authoring_free(opal);
    return rc;
}
