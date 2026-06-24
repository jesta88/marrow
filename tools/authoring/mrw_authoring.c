/* posix_memalign needs the POSIX feature macro under strict C11 (the target builds with
 * C_EXTENSIONS OFF → -std=c11, which otherwise hides the prototype on glibc). Must precede includes. */
#if !defined(_MSC_VER) && !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200112L
#endif

#include "mrw_authoring.h"
#include "marrow_internal.h" /* wire constants + section-type enums */

#include <math.h>            /* isfinite */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- aligned alloc (tool-owned; the runtime never allocates) ---- */
#if defined(_MSC_VER)
#  include <malloc.h>
void *mrw_authoring_alloc(size_t n) { return _aligned_malloc(n ? n : 1, 64); }
void  mrw_authoring_free(void *p)   { _aligned_free(p); }
#else
void *mrw_authoring_alloc(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n ? n : 1) != 0) p = NULL; return p; }
void  mrw_authoring_free(void *p)   { free(p); }
#endif

/* ---- LE writers ---- */
static void wr_u16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static void wr_u32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void wr_u64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
static void wr_f32(uint8_t *p, float v)    { memcpy(p, &v, 4); }
static void wr_id (uint8_t *p, mrw_id128 v) { wr_u64(p, v.lo); wr_u64(p + 8, v.hi); }

static uint32_t align_up(uint32_t x, uint32_t a) { return (x + (a - 1)) & ~(a - 1); }
static uint32_t lay(uint32_t *cur, uint32_t bytes) { *cur = align_up(*cur, 16); uint32_t off = *cur; *cur += bytes; return off; }

/* ---- float -> half (round to nearest even) ---- */
uint16_t mrw_authoring_f32_to_half(float f) {
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

/* ---- 64-bit overflow-checked size pass ----
 * Arbitrary authoring input can name counts that overflow the wire's offset/size fields, so the
 * layout is sized in 64-bit and each section is bounded to UINT32_MAX (its internal offsets and
 * the count fields are u32). Within that bound the byte core's u32 `lay()` is provably
 * non-wrapping. */
static int mul_ovf_u64(uint64_t a, uint64_t b, uint64_t *out) { if (a && b > UINT64_MAX / a) return 1; *out = a * b; return 0; }
static int add_ovf_u64(uint64_t a, uint64_t b, uint64_t *out) { if (a > UINT64_MAX - b) return 1; *out = a + b; return 0; }
static uint64_t align_up64(uint64_t x, uint64_t a) { return (x + (a - 1)) & ~(a - 1); }
static void     lay64(uint64_t *cur, uint64_t bytes) { *cur = align_up64(*cur, 16); *cur += bytes; }

static mrw_result name_blob_total_checked(const mrw_skel *s, uint64_t *out) {
    uint64_t total = 0;
    for (uint32_t j = 0; j < s->joint_count; ++j) {
        if (add_ovf_u64(total, (uint64_t)strlen(s->names[j]) + 1, &total)) return MRW_E_OVERFLOW;
    }
    if (total > UINT32_MAX) return MRW_E_OVERFLOW; /* nbs is a u32 wire field */
    *out = total;
    return MRW_OK;
}
static mrw_result skel_size_checked(const mrw_skel *s, uint64_t *out) {
    uint64_t nbs; mrw_result rc = name_blob_total_checked(s, &nbs); if (rc) return rc;
    uint64_t jc = s->joint_count, cur = 128;
    lay64(&cur, jc * 2); lay64(&cur, jc * 40); lay64(&cur, jc * 48); lay64(&cur, jc * 4); lay64(&cur, nbs);
    if (cur > UINT32_MAX) return MRW_E_OVERFLOW;
    *out = cur;
    return MRW_OK;
}
static mrw_result clip_size_checked(uint32_t jc, const mrw_clip *c, uint64_t *out) {
    uint64_t cur = 128, samples;
    uint32_t stride = c->codec ? 28u : 40u;   /* codec 1: q4+t3 (28 B); codec 0: q4+t3+s3 (40 B) */
    if (mul_ovf_u64((uint64_t)jc * c->sample_count, stride, &samples)) return MRW_E_OVERFLOW;
    lay64(&cur, samples);
    if (c->flags & MRW_CLIP_HAS_ROOT_MOTION) {
        uint64_t rt; if (mul_ovf_u64(c->sample_count, 28, &rt)) return MRW_E_OVERFLOW;
        lay64(&cur, rt);
    }
    if (cur > UINT32_MAX) return MRW_E_OVERFLOW;
    *out = cur;
    return MRW_OK;
}
static mrw_result baked_size_checked(const mrw_baked *b, uint32_t bone_count, uint64_t *out) {
    uint64_t fst = b->frame_stride_texels ? b->frame_stride_texels : (uint64_t)bone_count * 2;
    uint64_t texel_count;
    if (mul_ovf_u64(b->total_frames, fst, &texel_count)) return MRW_E_OVERFLOW;
    if (texel_count > UINT32_MAX) return MRW_E_OVERFLOW; /* texel_count is a u32 wire field */
    uint64_t cur = 128, ct, tx;
    if (mul_ovf_u64(b->clip_count, 48, &ct)) return MRW_E_OVERFLOW;
    lay64(&cur, ct);
    if (mul_ovf_u64(texel_count, 8, &tx)) return MRW_E_OVERFLOW;
    lay64(&cur, tx);
    if (cur > UINT32_MAX) return MRW_E_OVERFLOW;
    *out = cur;
    return MRW_OK;
}

/* ---- section sizing for the byte core (u32; safe - each section is bounded ≤ UINT32_MAX) ---- */
static uint32_t name_blob_total(const mrw_skel *s) {
    uint32_t total = 0;
    for (uint32_t j = 0; j < s->joint_count; ++j) total += (uint32_t)strlen(s->names[j]) + 1;
    return total;
}

/* ---- section writers; the byte-emitting core's output is pinned by test_golden_bytes.
 * write_skeleton is the only one that allocates, so it alone can fail (OOM). ---- */
static mrw_result write_skeleton(uint8_t *dst, const mrw_skel *s, mrw_id128 *out_id) {
    uint32_t jc = s->joint_count;
    uint32_t nbs = name_blob_total(s);
    uint32_t *name_off = (uint32_t *)malloc((size_t)jc * 4);
    uint8_t  *name_blob = (uint8_t *)malloc(nbs ? nbs : 1);
    if (!name_off || !name_blob) { free(name_off); free(name_blob); return MRW_E_OVERFLOW; }
    uint32_t acc = 0;
    for (uint32_t j = 0; j < jc; ++j) {
        name_off[j] = acc;
        uint32_t len = (uint32_t)strlen(s->names[j]) + 1;
        memcpy(name_blob + acc, s->names[j], len);
        acc += len;
    }
    mrw_id128 id;
    mrw_skeleton_compute_id(jc, nbs, s->parent, s->rest_local, s->inverse_bind, name_off, name_blob, &id);

    uint32_t cur = 128;
    uint32_t parent_off = lay(&cur, jc * 2);
    uint32_t rest_off   = lay(&cur, jc * 40);
    uint32_t ib_off     = lay(&cur, jc * 48);
    uint32_t noff_off   = lay(&cur, jc * 4);
    uint32_t nblob_off  = lay(&cur, nbs);
    memset(dst, 0, cur);

    wr_u32(dst + 0, jc);
    wr_u32(dst + 4, nbs);
    wr_id (dst + 8, id);
    wr_u32(dst + 24, parent_off);
    wr_u32(dst + 28, rest_off);
    wr_u32(dst + 32, ib_off);
    wr_u32(dst + 36, noff_off);
    wr_u32(dst + 40, nblob_off);
    wr_u32(dst + 44, 0);
    for (uint32_t j = 0; j < jc; ++j) wr_u16(dst + parent_off + j * 2, s->parent[j]);
    memcpy(dst + rest_off, s->rest_local, (size_t)jc * 40);
    memcpy(dst + ib_off,   s->inverse_bind, (size_t)jc * 48);
    for (uint32_t j = 0; j < jc; ++j) wr_u32(dst + noff_off + j * 4, name_off[j]);
    memcpy(dst + nblob_off, name_blob, nbs);

    free(name_off); free(name_blob);
    *out_id = id;
    return MRW_OK;
}

static mrw_id128 write_clip(uint8_t *dst, uint32_t jc, const mrw_clip *c, mrw_id128 skel_id) {
    mrw_id128 id;
    /* The clip id is over the canonical 10-float (q4+t3+s3) stream; for codec 1 the caller has snapped
     * every scale to exactly 1.0f, so this id equals the codec-1 canonical id (codec is not hashed). */
    mrw_clip_compute_id(jc, c->fps, c->sample_count, c->flags, c->samples, c->root_track, &id);
    int has_root = (c->flags & MRW_CLIP_HAS_ROOT_MOTION) != 0;
    uint32_t stride = c->codec ? 28u : 40u;   /* on-disk sample stride */
    uint64_t nsamp = (uint64_t)jc * c->sample_count;

    uint32_t cur = 128;
    uint32_t samples_off = lay(&cur, (uint32_t)(nsamp * stride));
    uint32_t root_off = has_root ? lay(&cur, c->sample_count * 28) : 0;
    memset(dst, 0, cur);

    wr_u32(dst + 0, jc);
    wr_u32(dst + 4, c->codec);
    wr_id (dst + 8, id);
    wr_id (dst + 24, skel_id);
    wr_f32(dst + 40, c->fps);
    wr_u32(dst + 44, c->sample_count);
    wr_u32(dst + 48, c->flags);
    wr_u32(dst + 52, samples_off);
    wr_u32(dst + 56, root_off);
    if (c->codec) {
        /* codec 1: emit only q4+t3 (first 28 B) of each 10-float source sample. */
        for (uint64_t r = 0; r < nsamp; ++r)
            memcpy(dst + samples_off + (size_t)r * 28, c->samples + (size_t)r * 10, 28);
    } else {
        memcpy(dst + samples_off, c->samples, (size_t)nsamp * 40);
    }
    if (has_root) memcpy(dst + root_off, c->root_track, (size_t)c->sample_count * 28);
    return id;
}

static void write_baked(uint8_t *dst, const mrw_baked *b, uint32_t bone_count,
                        mrw_id128 skel_id, const mrw_id128 *clip_ids) {
    uint32_t fst = b->frame_stride_texels ? b->frame_stride_texels : bone_count * 2;
    uint32_t texel_count = b->total_frames * fst;

    uint32_t cur = 128;
    uint32_t clip_table_off = lay(&cur, b->clip_count * 48);
    uint32_t texels_off = lay(&cur, texel_count * 8);
    memset(dst, 0, cur);

    wr_u32(dst + 0, 1);                  /* encoding 1 */
    wr_u32(dst + 4, bone_count);
    wr_u32(dst + 8, 2);                  /* texels_per_bone */
    wr_u32(dst + 12, fst);
    wr_u32(dst + 16, b->total_frames);
    wr_id (dst + 20, skel_id);
    wr_u32(dst + 36, b->clip_count);
    wr_u32(dst + 40, clip_table_off);
    wr_u32(dst + 44, texels_off);
    wr_u32(dst + 48, texel_count);
    wr_u32(dst + 52, 0);
    for (uint32_t c = 0; c < b->clip_count; ++c) {
        uint8_t *ce = dst + clip_table_off + (size_t)c * 48;
        wr_id (ce + 0, clip_ids[b->clip_index[c]]);
        wr_u32(ce + 16, b->first_frame[c]);
        wr_u32(ce + 20, b->frame_count[c]);
        wr_f32(ce + 24, b->source_duration[c]);
        wr_u32(ce + 28, b->clip_flags[c]);
    }
    for (uint32_t i = 0; i < texel_count * 4; ++i) wr_u16(dst + texels_off + i * 2, b->texels[i]);
}

/* ---- input validation (reject what the size pass / loader cannot represent) ---- */
static mrw_result validate(const mrw_skel *skel, const mrw_clip *clips, uint32_t nclip,
                           const mrw_baked *baked) {
    if (skel) {
        if (!skel->parent || !skel->rest_local || !skel->inverse_bind || !skel->names) return MRW_E_FORMAT;
        if (skel->joint_count < 1 || skel->joint_count > 0xFFFEu) return MRW_E_RANGE; /* joint-count range */
        for (uint32_t j = 0; j < skel->joint_count; ++j) if (!skel->names[j]) return MRW_E_FORMAT;
    }
    for (uint32_t c = 0; c < nclip; ++c) {
        const mrw_clip *cl = &clips[c];
        if (!cl->samples) return MRW_E_FORMAT;
        if (cl->sample_count < 1) return MRW_E_RANGE;
        if (!isfinite(cl->fps)) return MRW_E_RANGE;
        if (cl->sample_count >= 2 && !(cl->fps > 0.0f)) return MRW_E_RANGE; /* duration = (n-1)/fps */
        if ((cl->flags & MRW_CLIP_LOOPING) && cl->sample_count < 2) return MRW_E_RANGE; /* looping needs ≥2 samples */
        if ((cl->flags & MRW_CLIP_HAS_ROOT_MOTION) && !cl->root_track) return MRW_E_FORMAT;
        if (cl->codec > 1) return MRW_E_UNSUPPORTED;  /* only codec 0/1 are emittable */
    }
    if (baked) {
        if (baked->clip_count > 0 && (!baked->clip_index || !baked->first_frame || !baked->frame_count ||
                                      !baked->source_duration || !baked->clip_flags)) return MRW_E_FORMAT;
        for (uint32_t c = 0; c < baked->clip_count; ++c)
            if (baked->clip_index[c] >= nclip) return MRW_E_RANGE; /* write_baked indexes clip_ids[nclip] */
        uint32_t bone_count = skel ? skel->joint_count : 0;
        uint32_t fst = baked->frame_stride_texels ? baked->frame_stride_texels : bone_count * 2;
        if (baked->total_frames != 0 && fst != 0 && !baked->texels) return MRW_E_FORMAT; /* write_baked reads texels[] */
    }
    return MRW_OK;
}

mrw_result mrw_authoring_build(const mrw_skel *skel, const mrw_clip *clips, uint32_t nclip,
                               const mrw_baked *baked, uint8_t **out_buf, size_t *out_size) {
    if (!out_buf || !out_size) return MRW_E_FORMAT;
    if (nclip && !clips) return MRW_E_FORMAT;
    mrw_result rc = validate(skel, clips, nclip, baked);
    if (rc) return rc;

    uint32_t bone_count = skel ? skel->joint_count : 0;
    uint64_t nsec64 = (uint64_t)(skel ? 1u : 0u) + nclip + (baked ? 1u : 0u);
    if (nsec64 > UINT32_MAX) return MRW_E_OVERFLOW;
    uint32_t nsec = (uint32_t)nsec64;

    uint64_t *sec_off  = (uint64_t *)malloc((nsec ? nsec : 1) * sizeof(uint64_t));
    uint64_t *sec_sz   = (uint64_t *)malloc((nsec ? nsec : 1) * sizeof(uint64_t));
    uint32_t *sec_type = (uint32_t *)malloc((nsec ? nsec : 1) * sizeof(uint32_t));
    mrw_id128 *clip_ids = (mrw_id128 *)malloc((nclip ? nclip : 1) * sizeof(mrw_id128));
    uint8_t *buf = NULL;
    if (!sec_off || !sec_sz || !sec_type || !clip_ids) { rc = MRW_E_OVERFLOW; goto done; }

    /* ---- size pass: offsets in 64-bit, each section ≤ UINT32_MAX, total ≤ SIZE_MAX ---- */
    uint64_t table_off = MRW_FILE_HEADER_SIZE;
    uint64_t cur = table_off + (uint64_t)nsec * MRW_SECTION_ENTRY_SIZE;
    uint32_t si = 0;
    if (skel) {
        if ((rc = skel_size_checked(skel, &sec_sz[si]))) goto done;
        cur = align_up64(cur, MRW_ALIGN_SECTION); sec_off[si] = cur; sec_type[si] = MRW_SECTION_SKELETON;
        if (add_ovf_u64(cur, sec_sz[si], &cur)) { rc = MRW_E_OVERFLOW; goto done; }
        ++si;
    }
    for (uint32_t c = 0; c < nclip; ++c) {
        if ((rc = clip_size_checked(bone_count, &clips[c], &sec_sz[si]))) goto done;
        cur = align_up64(cur, MRW_ALIGN_SECTION); sec_off[si] = cur; sec_type[si] = MRW_SECTION_CLIP;
        if (add_ovf_u64(cur, sec_sz[si], &cur)) { rc = MRW_E_OVERFLOW; goto done; }
        ++si;
    }
    if (baked) {
        if ((rc = baked_size_checked(baked, bone_count, &sec_sz[si]))) goto done;
        cur = align_up64(cur, MRW_ALIGN_SECTION); sec_off[si] = cur; sec_type[si] = MRW_SECTION_BAKED;
        if (add_ovf_u64(cur, sec_sz[si], &cur)) { rc = MRW_E_OVERFLOW; goto done; }
        ++si;
    }
    uint64_t total = cur;
    if (total > (uint64_t)SIZE_MAX) { rc = MRW_E_OVERFLOW; goto done; }

    buf = (uint8_t *)mrw_authoring_alloc((size_t)total);
    if (!buf) { rc = MRW_E_OVERFLOW; goto done; } /* blob too large to allocate */
    memset(buf, 0, (size_t)total);

    /* ---- file header ---- */
    buf[0] = MRW_MAGIC0; buf[1] = MRW_MAGIC1; buf[2] = MRW_MAGIC2; buf[3] = MRW_MAGIC3;
    wr_u32(buf + 4, MRW_VERSION);
    wr_u32(buf + 8, MRW_ENDIAN_TAG);
    wr_u32(buf + 12, 0);
    wr_u32(buf + 16, nsec);
    wr_u32(buf + 20, (uint32_t)table_off);
    wr_u64(buf + 24, total);

    /* ---- sections + ids ---- */
    mrw_id128 skel_id = {0, 0};
    si = 0;
    if (skel) { if ((rc = write_skeleton(buf + sec_off[si], skel, &skel_id))) goto done; ++si; }
    for (uint32_t c = 0; c < nclip; ++c) { clip_ids[c] = write_clip(buf + sec_off[si], bone_count, &clips[c], skel_id); ++si; }
    if (baked) { write_baked(buf + sec_off[si], baked, bone_count, skel_id, clip_ids); ++si; }

    /* ---- section table ---- */
    for (uint32_t i = 0; i < nsec; ++i) {
        uint8_t *e = buf + table_off + (size_t)i * MRW_SECTION_ENTRY_SIZE;
        wr_u32(e + 0, sec_type[i]);
        wr_u32(e + 4, 0);
        wr_u64(e + 8, sec_off[i]);
        wr_u64(e + 16, sec_sz[i]);
        wr_u64(e + 24, 0);
    }

    *out_buf = buf;
    *out_size = (size_t)total;
    buf = NULL;        /* ownership transferred */
    rc = MRW_OK;

done:
    if (buf) mrw_authoring_free(buf);
    free(sec_off); free(sec_sz); free(sec_type); free(clip_ids);
    return rc;
}
