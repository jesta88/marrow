/* Baked GPU bake core. See mrw_bake.h for the contract and rationale. Reuses the runtime scalar
 * math; the shared sampling helpers (mrw_clampf / mrw_mod_pos / mrw_quat_nlerp) come from
 * marrow_internal.h so the component-space temporal interpolation here is bit-identical to the
 * runtime baked path (mrw_baked_sample_bone). Allocation-free: all buffers are caller-owned. The
 * f32→half encoder is a local static (not the authoring lib's, whose TU also defines the allocator)
 * so this core's symbol closure pulls in no malloc object - only the alloc-free mrw_decompose TU. */
#include "mrw_bake.h"
#include "marrow_internal.h"  /* mrw_clampf / mrw_mod_pos / mrw_quat_nlerp */

#include <math.h>
#include <stdint.h>
#include <string.h>

/* IEEE-754 binary32 → binary16 (round-to-nearest-even). Byte-identical to
 * mrw_authoring_f32_to_half (and the loader's inverse mrw_half_to_float); kept local so the bake
 * core does not link the authoring allocator TU. */
static uint16_t bake_f32_to_half(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t exp  = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp == 0xFF) { /* inf/nan */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u | (mant >> 13) : 0));
    }
    int e = (int)exp - 127 + 15;
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);          /* overflow -> inf */
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;                    /* underflow -> 0 */
        mant |= 0x800000u;                                     /* restore implicit 1 */
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t half_mant = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_mant & 1))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    uint32_t half_mant = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1))) {
        half_mant++;
        if (half_mant == 0x400u) { half_mant = 0; e++; if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | half_mant);
}

void mrw_bake_decode(const uint16_t *texels, uint32_t fst, uint32_t frame, uint32_t bone, mrw_xform *out) {
    const uint16_t *p = texels + ((size_t)frame * fst + (size_t)bone * 2) * 4;
    out->rot[0]=mrw_half_to_float(p[0]); out->rot[1]=mrw_half_to_float(p[1]);
    out->rot[2]=mrw_half_to_float(p[2]); out->rot[3]=mrw_half_to_float(p[3]);
    out->trans[0]=mrw_half_to_float(p[4]); out->trans[1]=mrw_half_to_float(p[5]);
    out->trans[2]=mrw_half_to_float(p[6]); out->scale=mrw_half_to_float(p[7]);
    float n2 = out->rot[0]*out->rot[0] + out->rot[1]*out->rot[1]
             + out->rot[2]*out->rot[2] + out->rot[3]*out->rot[3];
    float inv = (n2 > 0.0f) ? 1.0f / sqrtf(n2) : 0.0f;
    for (int k = 0; k < 4; ++k) out->rot[k] *= inv;
}

void mrw_bake_sample_xform(const uint16_t *texels, uint32_t fst,
                           uint32_t first_frame, uint32_t frame_count, float dur,
                           int looping, uint32_t bone, float t, mrw_xform *out) {
    uint32_t i0 = 0; float u = 0.0f;
    if (frame_count != 1 && dur != 0.0f) {
        float t_local = looping ? mrw_mod_pos(t, dur) : mrw_clampf(t, 0.0f, dur);
        float fpos = (t_local / dur) * (float)(frame_count - 1);
        uint32_t i = (uint32_t)floorf(fpos);
        if (i > frame_count - 2) i = frame_count - 2;
        i0 = i; u = fpos - (float)i;
    }
    mrw_xform a; mrw_bake_decode(texels, fst, first_frame + i0, bone, &a);
    if (u == 0.0f) { *out = a; return; }
    mrw_xform b; mrw_bake_decode(texels, fst, first_frame + i0 + 1, bone, &b);
    mrw_quat_nlerp(a.rot, b.rot, u, out->rot);
    for (int k = 0; k < 3; ++k) out->trans[k] = a.trans[k] + u*(b.trans[k]-a.trans[k]);
    out->scale = a.scale + u*(b.scale - a.scale);
}

void mrw_xform_nlerp(const mrw_xform *a, const mrw_xform *b, float w, mrw_xform *out) {
    mrw_quat_nlerp(a->rot, b->rot, w, out->rot);
    for (int k = 0; k < 3; ++k) out->trans[k] = a->trans[k] + w*(b->trans[k]-a->trans[k]);
    out->scale = a->scale + w*(b->scale - a->scale);
}

/* scratch layout (all regions 16-aligned; sizes are 16-multiples for any joint_count):
 *   model   : joint_count·12 floats   (mrw_clip_to_palette scratch_model)
 *   palette : joint_count·12 floats   (mrw_clip_to_palette out_palette)
 *   prevq   : joint_count·4  floats   (last frame's quat, for sign continuity)                */
#define BAKE_SCRATCH_FLOATS_PER_JOINT (12u + 12u + 4u)

static int mul_ovf_u64(uint64_t a, uint64_t b, uint64_t *out) { if (a && b > UINT64_MAX / a) return 1; *out = a * b; return 0; }

mrw_result mrw_bake_clip_requirements(uint32_t joint_count, uint32_t frame_count,
                                      mrw_mem_req *out_scratch, mrw_mem_req *out_texels) {
    if (frame_count == 0) return MRW_E_RANGE;

    /* Genuinely overflow-checked 64-bit products, then a SIZE_MAX bound - overflow ⇒ MRW_E_OVERFLOW.
     * Each multiply is checked so arbitrary joint_count·frame_count cannot wrap. */
    uint64_t jc = joint_count;
    uint64_t scratch_bytes, texel_bytes;
    if (mul_ovf_u64(jc, BAKE_SCRATCH_FLOATS_PER_JOINT * (uint64_t)sizeof(float), &scratch_bytes)) return MRW_E_OVERFLOW;
    if (mul_ovf_u64((uint64_t)frame_count, jc, &texel_bytes)) return MRW_E_OVERFLOW;
    if (mul_ovf_u64(texel_bytes, 16u /* 2 texels · 8 bytes (RGBA16F) */, &texel_bytes)) return MRW_E_OVERFLOW;
    if (scratch_bytes > (uint64_t)SIZE_MAX || texel_bytes > (uint64_t)SIZE_MAX) return MRW_E_OVERFLOW;

    if (out_scratch) { out_scratch->size = (size_t)scratch_bytes; out_scratch->align = 16; }
    if (out_texels)  { out_texels->size  = (size_t)texel_bytes;   out_texels->align  = 16; }
    return MRW_OK;
}

mrw_result mrw_bake_clip(const mrw_skeleton_view *skel, const mrw_clip_view *clip,
                         uint32_t frame_count, const uint32_t *probe_counts, const float *probes,
                         float tol, void *scratch, size_t scratch_capacity,
                         uint16_t *out_texels, size_t out_texels_capacity, mrw_bake_stats *out_stats) {
    mrw_bake_stats st = { 0, INFINITY, 0, 0, MRW_BAKE_STRUCTURAL };
    if (out_stats) *out_stats = st;

    /* argument validation: the header promises an error (not UB) for bad args. */
    if (!skel || !clip || !scratch || !out_texels) return MRW_E_FORMAT;

    uint32_t bc  = skel->joint_count;
    uint32_t fst = bc * 2;

    /* probes==NULL is allowed iff every count is 0 (a bone with probes needs a probe array). */
    if (probe_counts && !probes)
        for (uint32_t b = 0; b < bc; ++b) if (probe_counts[b] > 0) return MRW_E_FORMAT;

    /* caller-owned-buffer contract: size, then alignment. */
    mrw_mem_req need_scratch, need_texels;
    mrw_result rr = mrw_bake_clip_requirements(bc, frame_count, &need_scratch, &need_texels);
    if (rr != MRW_OK) return rr;                                    /* frame_count==0 / overflow */
    if (scratch_capacity < need_scratch.size) return MRW_E_CAPACITY;
    if (out_texels_capacity < need_texels.size) return MRW_E_CAPACITY;
    if (((uintptr_t)scratch & 15u) != 0u) return MRW_E_ALIGN;
    if (((uintptr_t)out_texels & 15u) != 0u) return MRW_E_ALIGN;

    float *model   = (float *)scratch;
    float *palette = model + (size_t)bc * 12;
    float *prevq   = palette + (size_t)bc * 12;

    float dur = (clip->sample_count <= 1) ? 0.0f : (float)(clip->sample_count - 1) / clip->fps;

    int             eligible    = 1;
    float           worst       = 0.0f;            /* max probe displacement (m); +INF if structural */
    uint32_t        worst_bone  = 0, worst_frame = 0;
    mrw_bake_reason worst_reason = MRW_BAKE_OK;
    int             have_prev   = 0;

    for (uint32_t f = 0; f < frame_count; ++f) {
        float t_f = (frame_count < 2) ? 0.0f : ((float)f / (float)(frame_count - 1)) * dur;
        mrw_result pr = mrw_clip_to_palette(skel, clip, t_f, model, palette, bc);
        if (pr != MRW_OK) {
            if (out_stats) { st.worst_frame = f; *out_stats = st; }
            return pr;                                              /* CPU sampling failure */
        }

        size_t poff = 0;                                           /* prefix offset into probes[] */
        for (uint32_t b = 0; b < bc; ++b) {
            const float *M = palette + (size_t)b * 12;
            mrw_xform x;
            int ok = mrw_decompose_affine(M, &x);  /* x = identity on structural failure */
            uint32_t np_b = probe_counts ? probe_counts[b] : 0;

            int             bone_ok = 1;
            mrw_bake_reason r       = MRW_BAKE_OK;
            float           cand    = 0.0f;        /* this bone's contribution to `worst` */

            /* The structural reject applies to EVERY bone, probed or not (a non-decomposable
             * palette would bake a meaningless identity). The perceptual residual is separate: a
             * bone with NO probes (np==0) is eligible by default for it (it deforms nothing visible)
             * - skipping the residual also keeps `probes` un-indexed, so a NULL probe array is
             * well-defined there. */
            if (!ok) {
                cand = INFINITY; r = MRW_BAKE_STRUCTURAL; bone_ok = 0;
            } else if (np_b > 0) {
                float mp[12]; mrw_xform_to_affine(&x, mp);
                float resid = mrw_affine_probe_dist(M, mp, np_b, probes + poff);
                cand = resid;
                if (!(resid <= tol)) { r = MRW_BAKE_RESIDUAL; bone_ok = 0; }
            }
            poff += (size_t)np_b * 3;

            /* Sign-continuity: keep the temporal quaternion track on one hemisphere. */
            if (have_prev) {
                float dot = x.rot[0]*prevq[b*4+0] + x.rot[1]*prevq[b*4+1]
                          + x.rot[2]*prevq[b*4+2] + x.rot[3]*prevq[b*4+3];
                if (dot < 0.0f) for (int k = 0; k < 4; ++k) x.rot[k] = -x.rot[k];
            }
            for (int k = 0; k < 4; ++k) prevq[b*4+k] = x.rot[k];

            uint16_t *p = out_texels + ((size_t)f * fst + (size_t)b * 2) * 4;
            p[0]=bake_f32_to_half(x.rot[0]);   p[1]=bake_f32_to_half(x.rot[1]);
            p[2]=bake_f32_to_half(x.rot[2]);   p[3]=bake_f32_to_half(x.rot[3]);
            p[4]=bake_f32_to_half(x.trans[0]); p[5]=bake_f32_to_half(x.trans[1]);
            p[6]=bake_f32_to_half(x.trans[2]); p[7]=bake_f32_to_half(x.scale);
            /* A half overflow (e.g. |t| > 65504) yields an inf texel the loader rejects. Check ALL
             * bones (probed or not): the loader validates every texel, so an unprobed bone must
             * still emit a finite stream. A non-finite texel rejects the rig. */
            for (int k = 0; k < 8; ++k) {
                if (!isfinite(mrw_half_to_float(p[k]))) {
                    bone_ok = 0;
                    if (r == MRW_BAKE_OK) r = MRW_BAKE_QUANTIZED;
                    if (cand < INFINITY) cand = INFINITY;
                    break;
                }
            }

            if (!bone_ok) eligible = 0;
            if (cand > worst) { worst = cand; worst_bone = b; worst_frame = f; worst_reason = r; }
        }
        have_prev = 1;
    }

    if (out_stats) {
        out_stats->eligible     = eligible;
        out_stats->max_residual = worst;
        out_stats->worst_bone   = worst_bone;
        out_stats->worst_frame  = worst_frame;
        out_stats->reason       = eligible ? MRW_BAKE_OK : worst_reason;
    }
    return MRW_OK;
}
