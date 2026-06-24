/* marrow .mrw loader - validate, then view. Header/metadata are read via
 * memcpy into native values (mrw_rd_*, never struct overlay); bulk arrays are accessed
 * in place. mrw_blob_open performs the ENTIRE deterministic validation pass; the
 * accessors below are then pure, pre-checked locators. */
#include "marrow_internal.h"

/* ------------------------------------------------------------------ range helpers */

/* In-section array span check: 16-aligned, after the 128-byte section header, and the
 * whole [off, off+bytes) range inside the section. `bytes` is precomputed by the caller
 * (overflow-safe in u64; section size < 2^32). */
static mrw_result span_ok(uint32_t off, uint64_t bytes, uint32_t sec_size) {
    if (off % MRW_ALIGN_ARRAY != 0) return MRW_E_ALIGN;
    if (off < MRW_SECTION_HEADER_SIZE) return MRW_E_RANGE;
    if ((uint64_t)off + bytes > (uint64_t)sec_size) return MRW_E_RANGE;
    return MRW_OK;
}

/* check `n` consecutive f32 finite; if `quat`, also that the leading 4 are near-unit */
static mrw_result floats_ok(const uint8_t *p, uint32_t n, int quat, float quat_tol) {
    for (uint32_t i = 0; i < n; ++i)
        if (!mrw_f32_finite(mrw_rd_f32(p + (size_t)i * 4))) return MRW_E_RANGE;
    if (quat) {
        float x = mrw_rd_f32(p + 0), y = mrw_rd_f32(p + 4), z = mrw_rd_f32(p + 8), w = mrw_rd_f32(p + 12);
        if (!mrw_quat_near_unit(x, y, z, w, quat_tol)) return MRW_E_FORMAT;
    }
    return MRW_OK;
}

static int bytes_all_zero(const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; ++i) if (p[i] != 0) return 0;
    return 1;
}

/* ------------------------------------------------------------------ section locators */

static const uint8_t *section_entry(const uint8_t *base, uint32_t st_off, uint32_t i) {
    return base + (size_t)st_off + (size_t)i * MRW_SECTION_ENTRY_SIZE;
}

/* Locate a CLIP by id among already-validated sections; report its timing/flags. */
static int find_clip(const uint8_t *base, uint32_t st_off, uint32_t section_count,
                     const mrw_id128 *want, uint32_t *out_sample_count, float *out_fps, uint32_t *out_flags) {
    for (uint32_t i = 0; i < section_count; ++i) {
        const uint8_t *e = section_entry(base, st_off, i);
        if (mrw_rd_u32(e) != MRW_SECTION_CLIP) continue;
        const uint8_t *h = base + mrw_rd_u64(e + 8);
        mrw_id128 cid = mrw_rd_id128(h + 8);
        if (mrw_id_equal(&cid, want)) {
            *out_sample_count = mrw_rd_u32(h + 44);
            *out_fps = mrw_rd_f32(h + 40);
            *out_flags = mrw_rd_u32(h + 48);
            return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ per-section validation */

static mrw_result validate_skeleton(const uint8_t *base, uint64_t off, uint64_t sz,
                                       mrw_id128 *out_id, uint32_t *out_joints) {
    if (sz < MRW_SECTION_HEADER_SIZE) return MRW_E_FORMAT;
    const uint8_t *h = base + off;
    uint32_t joint_count    = mrw_rd_u32(h + 0);
    uint32_t name_blob_size = mrw_rd_u32(h + 4);
    mrw_id128 id         = mrw_rd_id128(h + 8);
    uint32_t parent_off     = mrw_rd_u32(h + 24);
    uint32_t rest_off       = mrw_rd_u32(h + 28);
    uint32_t ib_off         = mrw_rd_u32(h + 32);
    uint32_t noff_off       = mrw_rd_u32(h + 36);
    uint32_t nblob_off      = mrw_rd_u32(h + 40);
    uint32_t flags          = mrw_rd_u32(h + 44);
    if (flags != 0) return MRW_E_FORMAT;
    if (!bytes_all_zero(h + 48, 80)) return MRW_E_FORMAT;
    if (joint_count < 1 || joint_count > 0xFFFE) return MRW_E_RANGE;
    uint32_t secsz = (uint32_t)sz;
    mrw_result rc;
    if ((rc = span_ok(parent_off, (uint64_t)joint_count * MRW_PARENT_STRIDE,       secsz))) return rc;
    if ((rc = span_ok(rest_off,   (uint64_t)joint_count * MRW_REST_LOCAL_STRIDE,   secsz))) return rc;
    if ((rc = span_ok(ib_off,     (uint64_t)joint_count * MRW_INVERSE_BIND_STRIDE, secsz))) return rc;
    if ((rc = span_ok(noff_off,   (uint64_t)joint_count * MRW_NAME_OFF_STRIDE,     secsz))) return rc;
    if ((rc = span_ok(nblob_off,  (uint64_t)name_blob_size,                           secsz))) return rc;

    /* parents: parent[0]==0xFFFF (single root); parent[j]<j for j>=1; no other 0xFFFF */
    for (uint32_t j = 0; j < joint_count; ++j) {
        uint16_t pj = mrw_rd_u16(h + parent_off + (size_t)j * MRW_PARENT_STRIDE);
        if (j == 0) { if (pj != 0xFFFF) return MRW_E_FORMAT; }
        else { if (pj == 0xFFFF) return MRW_E_FORMAT; if (pj >= j) return MRW_E_RANGE; }
    }
    /* rest_local: 10 f32/joint, quat (first 4) near-unit; inverse_bind: 12 f32/joint */
    for (uint32_t j = 0; j < joint_count; ++j) {
        if ((rc = floats_ok(h + rest_off + (size_t)j * MRW_REST_LOCAL_STRIDE, 10, 1, 1e-3f))) return rc;
        if ((rc = floats_ok(h + ib_off + (size_t)j * MRW_INVERSE_BIND_STRIDE, 12, 0, 0))) return rc;
    }
    /* names: name_off[j] < name_blob_size and a NUL terminator exists within the blob */
    for (uint32_t j = 0; j < joint_count; ++j) {
        uint32_t no = mrw_rd_u32(h + noff_off + (size_t)j * MRW_NAME_OFF_STRIDE);
        if (no >= name_blob_size) return MRW_E_RANGE;
        const uint8_t *nb = h + nblob_off;
        uint32_t k = no, found = 0;
        for (; k < name_blob_size; ++k) if (nb[k] == 0) { found = 1; break; }
        if (!found) return MRW_E_FORMAT;
    }
    *out_id = id; *out_joints = joint_count;
    return MRW_OK;
}

static mrw_result validate_clip(const uint8_t *base, uint64_t off, uint64_t sz,
                                   int skel_present, const mrw_id128 *skel_id, uint32_t skel_joints) {
    if (sz < MRW_SECTION_HEADER_SIZE) return MRW_E_FORMAT;
    const uint8_t *h = base + off;
    uint32_t joint_count  = mrw_rd_u32(h + 0);
    uint32_t codec        = mrw_rd_u32(h + 4);
    mrw_id128 skeleton_id = mrw_rd_id128(h + 24);
    float    fps          = mrw_rd_f32(h + 40);
    uint32_t sample_count = mrw_rd_u32(h + 44);
    uint32_t flags        = mrw_rd_u32(h + 48);
    uint32_t samples_off  = mrw_rd_u32(h + 52);
    uint32_t root_off     = mrw_rd_u32(h + 56);
    if (!bytes_all_zero(h + 60, 68)) return MRW_E_FORMAT;
    /* codec 0 = raw fixed-rate TRS (40 B/sample, q4+t3+s3); codec 1 = scale-free, 28 B/sample (q4+t3),
     * scale implicitly (1,1,1). The on-disk sample stride is DERIVED from the codec - there is no
     * stored stride field. Any other codec → MRW_E_UNSUPPORTED (blob stays well-formed). */
    if (codec > 1) return MRW_E_UNSUPPORTED;
    uint32_t sample_stride = codec ? MRW_ROOT_SAMPLE_STRIDE : MRW_CLIP_SAMPLE_STRIDE; /* 28 : 40 */
    uint32_t sample_floats = codec ? 7u : 10u;                                        /* q4+t3 : q4+t3+s3 */
    if (flags & ~(uint32_t)(MRW_CLIP_LOOPING | MRW_CLIP_HAS_ROOT_MOTION)) return MRW_E_FORMAT;
    if (!skel_present) return MRW_E_INCOMPATIBLE;
    if (!mrw_id_equal(&skeleton_id, skel_id)) return MRW_E_INCOMPATIBLE;
    if (joint_count != skel_joints) return MRW_E_INCOMPATIBLE;
    if (!mrw_f32_finite(fps) || fps <= 0.0f) return MRW_E_RANGE;
    if (sample_count < 1) return MRW_E_RANGE;
    if ((flags & MRW_CLIP_LOOPING) && sample_count < 2) return MRW_E_FORMAT;

    uint32_t secsz = (uint32_t)sz;
    mrw_result rc;
    uint64_t nsamp = (uint64_t)joint_count * sample_count;
    if ((rc = span_ok(samples_off, nsamp * sample_stride, secsz))) return rc;
    int has_root = (flags & MRW_CLIP_HAS_ROOT_MOTION) != 0;
    if (has_root) {
        if (root_off == 0) return MRW_E_FORMAT;
        if ((rc = span_ok(root_off, (uint64_t)sample_count * MRW_ROOT_SAMPLE_STRIDE, secsz))) return rc;
    } else {
        if (root_off != 0) return MRW_E_FORMAT;
    }
    for (uint64_t r = 0; r < nsamp; ++r)
        if ((rc = floats_ok(h + samples_off + r * sample_stride, sample_floats, 1, 1e-3f))) return rc;
    if (has_root)
        for (uint32_t s = 0; s < sample_count; ++s)
            if ((rc = floats_ok(h + root_off + (size_t)s * MRW_ROOT_SAMPLE_STRIDE, 7, 1, 1e-3f))) return rc;
    return MRW_OK;
}

static mrw_result validate_baked(const uint8_t *base, uint32_t st_off, uint32_t section_count,
                                    uint64_t off, uint64_t sz,
                                    int skel_present, const mrw_id128 *skel_id, uint32_t skel_joints) {
    if (sz < MRW_SECTION_HEADER_SIZE) return MRW_E_FORMAT;
    const uint8_t *h = base + off;
    uint32_t encoding       = mrw_rd_u32(h + 0);
    uint32_t bone_count     = mrw_rd_u32(h + 4);
    uint32_t texels_per_bone= mrw_rd_u32(h + 8);
    uint32_t frame_stride   = mrw_rd_u32(h + 12);
    uint32_t total_frames   = mrw_rd_u32(h + 16);
    mrw_id128 skeleton_id= mrw_rd_id128(h + 20);
    uint32_t clip_count     = mrw_rd_u32(h + 36);
    uint32_t clip_table_off = mrw_rd_u32(h + 40);
    uint32_t texels_off     = mrw_rd_u32(h + 44);
    uint32_t texel_count    = mrw_rd_u32(h + 48);
    uint32_t flags          = mrw_rd_u32(h + 52);
    if (!bytes_all_zero(h + 56, 72)) return MRW_E_FORMAT;
    if (flags != 0) return MRW_E_FORMAT;
    if (encoding != 1) return MRW_E_UNSUPPORTED;
    if (texels_per_bone != 2) return MRW_E_FORMAT;
    if (!skel_present) return MRW_E_INCOMPATIBLE;
    if (!mrw_id_equal(&skeleton_id, skel_id)) return MRW_E_INCOMPATIBLE;
    if (bone_count != skel_joints) return MRW_E_INCOMPATIBLE;
    if (frame_stride < bone_count * 2u) return MRW_E_FORMAT;
    /* texel_count == total_frames · frame_stride (checked equality, overflow-safe) */
    if ((uint64_t)total_frames * (uint64_t)frame_stride != (uint64_t)texel_count) return MRW_E_FORMAT;

    uint32_t secsz = (uint32_t)sz;
    mrw_result rc;
    if ((rc = span_ok(clip_table_off, (uint64_t)clip_count * MRW_BAKED_CLIP_STRIDE, secsz))) return rc;
    if ((rc = span_ok(texels_off, (uint64_t)texel_count * MRW_TEXEL_STRIDE, secsz))) return rc;

    for (uint32_t c = 0; c < clip_count; ++c) {
        const uint8_t *ce = h + clip_table_off + (size_t)c * MRW_BAKED_CLIP_STRIDE;
        mrw_id128 clip_id = mrw_rd_id128(ce + 0);
        uint32_t first_frame = mrw_rd_u32(ce + 16);
        uint32_t frame_count = mrw_rd_u32(ce + 20);
        float    src_dur     = mrw_rd_f32(ce + 24);
        uint32_t cflags      = mrw_rd_u32(ce + 28);
        if (!bytes_all_zero(ce + 32, 16)) return MRW_E_FORMAT;
        if (cflags & ~(uint32_t)MRW_BAKED_CLIP_LOOPING) return MRW_E_FORMAT;
        if (frame_count < 1) return MRW_E_RANGE;
        if ((uint64_t)first_frame + frame_count > (uint64_t)total_frames) return MRW_E_RANGE;

        uint32_t cl_sc, cl_flags; float cl_fps;
        if (!find_clip(base, st_off, section_count, &clip_id, &cl_sc, &cl_fps, &cl_flags))
            return MRW_E_INCOMPATIBLE;
        if (!mrw_f32_finite(src_dur) || src_dur < 0.0f) return MRW_E_RANGE;
        /* source_duration_s MUST equal the paired CLIP duration, AND be 0 iff frame_count==1.
         * Checking the equality unconditionally rejects a one-frame baked cache (dur 0) that
         * points at a dynamic clip (dur > 0). */
        float clip_dur = (cl_sc <= 1) ? 0.0f : (float)(cl_sc - 1) / cl_fps;
        if (frame_count == 1) { if (src_dur != 0.0f) return MRW_E_FORMAT; }
        else                  { if (!(src_dur > 0.0f)) return MRW_E_FORMAT; }
        if (src_dur != clip_dur) return MRW_E_FORMAT;
        int baked_loop = (cflags & MRW_BAKED_CLIP_LOOPING) != 0;
        int clip_loop  = (cl_flags & MRW_CLIP_LOOPING) != 0;
        if (baked_loop != clip_loop) return MRW_E_FORMAT;
    }

    /* all texels finite; each decoded bone quaternion near-unit (1e-2 headroom) */
    for (uint32_t i = 0; i < texel_count; ++i) {
        const uint8_t *tx = h + texels_off + (size_t)i * MRW_TEXEL_STRIDE;
        for (int c = 0; c < 4; ++c)
            if (!mrw_f32_finite(mrw_half_to_float(mrw_rd_u16(tx + c * 2)))) return MRW_E_RANGE;
    }
    for (uint32_t f = 0; f < total_frames; ++f) {
        for (uint32_t b = 0; b < bone_count; ++b) {
            const uint8_t *t0 = h + texels_off + ((size_t)f * frame_stride + (size_t)b * texels_per_bone) * MRW_TEXEL_STRIDE;
            float qx = mrw_half_to_float(mrw_rd_u16(t0 + 0));
            float qy = mrw_half_to_float(mrw_rd_u16(t0 + 2));
            float qz = mrw_half_to_float(mrw_rd_u16(t0 + 4));
            float qw = mrw_half_to_float(mrw_rd_u16(t0 + 6));
            if (!mrw_quat_near_unit(qx, qy, qz, qw, 1e-2f)) return MRW_E_FORMAT;
        }
    }
    return MRW_OK;
}

/* ------------------------------------------------------------------ open (full validation) */

mrw_result mrw_blob_open(const void *data, uint64_t size, mrw_blob *out) {
    if (!data || !out) return MRW_E_RANGE;
    const uint8_t *base = (const uint8_t *)data;
    if (((uintptr_t)base & (MRW_ALIGN_SECTION - 1)) != 0) return MRW_E_ALIGN;
    if (size < MRW_FILE_HEADER_SIZE) return MRW_E_FORMAT;

    if (base[0] != MRW_MAGIC0 || base[1] != MRW_MAGIC1 ||
        base[2] != MRW_MAGIC2 || base[3] != MRW_MAGIC3) return MRW_E_FORMAT;
    if (mrw_rd_u32(base + 4) != MRW_VERSION) return MRW_E_FORMAT;
    if (mrw_rd_u32(base + 8) != MRW_ENDIAN_TAG) return MRW_E_FORMAT;
    if (mrw_rd_u32(base + 12) != 0) return MRW_E_FORMAT;
    uint32_t section_count = mrw_rd_u32(base + 16);
    uint32_t st_off        = mrw_rd_u32(base + 20);
    uint64_t blob_size     = mrw_rd_u64(base + 24);
    if (blob_size != size) return MRW_E_FORMAT;
    if (!bytes_all_zero(base + 32, 32)) return MRW_E_FORMAT;

    uint64_t table_bytes = (uint64_t)section_count * MRW_SECTION_ENTRY_SIZE;
    /* section_table_off is 64-aligned and in-blob regardless of section_count. */
    if (st_off % MRW_ALIGN_SECTION != 0) return MRW_E_ALIGN;
    if ((uint64_t)st_off > blob_size || table_bytes > blob_size - st_off) return MRW_E_RANGE;
    if (section_count > 0 && st_off < MRW_FILE_HEADER_SIZE) return MRW_E_FORMAT;

    /* Pass 1: structural section-table validation (sorted, in-blob, non-overlap), and
     * locate the single optional SKELETON. */
    uint64_t prev_end = 0;
    int skel_present = 0; uint64_t skel_off = 0, skel_sz = 0;
    for (uint32_t i = 0; i < section_count; ++i) {
        const uint8_t *e = section_entry(base, st_off, i);
        uint32_t type  = mrw_rd_u32(e + 0);
        uint32_t sflags= mrw_rd_u32(e + 4);
        uint64_t off   = mrw_rd_u64(e + 8);
        uint64_t sz    = mrw_rd_u64(e + 16);
        uint64_t rsv   = mrw_rd_u64(e + 24);
        if (rsv != 0) return MRW_E_FORMAT;
        if (sflags & ~(uint32_t)MRW_SECTION_FLAG_OPTIONAL) return MRW_E_FORMAT;
        if (off % MRW_ALIGN_SECTION != 0) return MRW_E_ALIGN;
        if (off < MRW_FILE_HEADER_SIZE) return MRW_E_FORMAT;
        if (off > blob_size || sz > blob_size - off) return MRW_E_RANGE;
        if (sz >= (1ull << 32)) return MRW_E_RANGE;              /* must stay u32-addressable */
        if (off < prev_end) return MRW_E_FORMAT;                 /* sorted + non-overlapping */
        if (!(off + sz <= st_off || (uint64_t)st_off + table_bytes <= off)) return MRW_E_FORMAT; /* vs table */
        prev_end = off + sz;

        if (type == MRW_SECTION_SKELETON) {
            if (skel_present) return MRW_E_FORMAT;               /* at most one SKELETON per blob */
            skel_present = 1; skel_off = off; skel_sz = sz;
        } else if (type != MRW_SECTION_CLIP && type != MRW_SECTION_BAKED) {
            if (!(sflags & MRW_SECTION_FLAG_OPTIONAL)) return MRW_E_FORMAT; /* unknown, not skippable */
        }
    }

    mrw_id128 skel_id; uint32_t skel_joints = 0;
    mrw_result rc;
    if (skel_present) {
        if ((rc = validate_skeleton(base, skel_off, skel_sz, &skel_id, &skel_joints))) return rc;
    }

    /* Pass 2: CLIP sections (need the skeleton). */
    for (uint32_t i = 0; i < section_count; ++i) {
        const uint8_t *e = section_entry(base, st_off, i);
        if (mrw_rd_u32(e) != MRW_SECTION_CLIP) continue;
        if ((rc = validate_clip(base, mrw_rd_u64(e + 8), mrw_rd_u64(e + 16),
                                skel_present, &skel_id, skel_joints))) return rc;
    }

    /* Pass 3: BAKED sections (need skeleton + clips). */
    for (uint32_t i = 0; i < section_count; ++i) {
        const uint8_t *e = section_entry(base, st_off, i);
        if (mrw_rd_u32(e) != MRW_SECTION_BAKED) continue;
        if ((rc = validate_baked(base, st_off, section_count, mrw_rd_u64(e + 8), mrw_rd_u64(e + 16),
                                 skel_present, &skel_id, skel_joints))) return rc;
    }

    out->base = base;
    out->size = size;
    out->section_table = (section_count > 0) ? base + st_off : NULL;
    out->section_count = section_count;
    out->section_table_off = st_off;
    return MRW_OK;
}

/* ------------------------------------------------------------------ accessors */

mrw_result mrw_blob_section_type(const mrw_blob *b, uint32_t index, uint32_t *out_type) {
    if (!b || !out_type) return MRW_E_RANGE;
    if (index >= b->section_count) return MRW_E_RANGE;
    *out_type = mrw_rd_u32(section_entry(b->base, b->section_table_off, index));
    return MRW_OK;
}

static mrw_result section_of_type(const mrw_blob *b, uint32_t index, uint32_t type, uint64_t *out_off) {
    if (!b) return MRW_E_RANGE;
    if (index >= b->section_count) return MRW_E_RANGE;
    const uint8_t *e = section_entry(b->base, b->section_table_off, index);
    if (mrw_rd_u32(e) != type) return MRW_E_INCOMPATIBLE;
    *out_off = mrw_rd_u64(e + 8);
    return MRW_OK;
}

static void fill_skeleton(const uint8_t *h, mrw_skeleton_view *v) {
    v->base = h;
    v->joint_count    = mrw_rd_u32(h + 0);
    v->name_blob_size = mrw_rd_u32(h + 4);
    v->id             = mrw_rd_id128(h + 8);
    v->parent_off       = mrw_rd_u32(h + 24);
    v->rest_local_off   = mrw_rd_u32(h + 28);
    v->inverse_bind_off = mrw_rd_u32(h + 32);
    v->name_off_off     = mrw_rd_u32(h + 36);
    v->name_blob_off    = mrw_rd_u32(h + 40);
}

static void fill_clip(const uint8_t *h, mrw_clip_view *v) {
    v->base = h;
    v->joint_count  = mrw_rd_u32(h + 0);
    v->codec        = mrw_rd_u32(h + 4);
    v->id           = mrw_rd_id128(h + 8);
    v->skeleton_id  = mrw_rd_id128(h + 24);
    v->fps          = mrw_rd_f32(h + 40);
    v->sample_count = mrw_rd_u32(h + 44);
    v->flags        = mrw_rd_u32(h + 48);
    v->samples_off  = mrw_rd_u32(h + 52);
    v->root_track_off = mrw_rd_u32(h + 56);
}

static void fill_baked(const uint8_t *h, mrw_baked_view *v) {
    v->base = h;
    v->encoding           = mrw_rd_u32(h + 0);
    v->bone_count         = mrw_rd_u32(h + 4);
    v->texels_per_bone    = mrw_rd_u32(h + 8);
    v->frame_stride_texels= mrw_rd_u32(h + 12);
    v->total_frames       = mrw_rd_u32(h + 16);
    v->skeleton_id        = mrw_rd_id128(h + 20);
    v->clip_count         = mrw_rd_u32(h + 36);
    v->clip_table_off     = mrw_rd_u32(h + 40);
    v->texels_off         = mrw_rd_u32(h + 44);
    v->texel_count        = mrw_rd_u32(h + 48);
    v->flags              = mrw_rd_u32(h + 52);
}

mrw_result mrw_skeleton_view_at(const mrw_blob *b, uint32_t index, mrw_skeleton_view *out) {
    if (!out) return MRW_E_RANGE;
    uint64_t off; mrw_result rc = section_of_type(b, index, MRW_SECTION_SKELETON, &off);
    if (rc) return rc;
    fill_skeleton(b->base + off, out);
    return MRW_OK;
}
mrw_result mrw_clip_view_at(const mrw_blob *b, uint32_t index, mrw_clip_view *out) {
    if (!out) return MRW_E_RANGE;
    uint64_t off; mrw_result rc = section_of_type(b, index, MRW_SECTION_CLIP, &off);
    if (rc) return rc;
    fill_clip(b->base + off, out);
    return MRW_OK;
}
mrw_result mrw_baked_view_at(const mrw_blob *b, uint32_t index, mrw_baked_view *out) {
    if (!out) return MRW_E_RANGE;
    uint64_t off; mrw_result rc = section_of_type(b, index, MRW_SECTION_BAKED, &off);
    if (rc) return rc;
    fill_baked(b->base + off, out);
    return MRW_OK;
}

mrw_result mrw_blob_skeleton(const mrw_blob *b, mrw_skeleton_view *out) {
    if (!b || !out) return MRW_E_RANGE;
    uint32_t st_off = b->section_table_off;
    for (uint32_t i = 0; i < b->section_count; ++i) {
        const uint8_t *e = section_entry(b->base, st_off, i);
        if (mrw_rd_u32(e) == MRW_SECTION_SKELETON) { fill_skeleton(b->base + mrw_rd_u64(e + 8), out); return MRW_OK; }
    }
    return MRW_E_INCOMPATIBLE;
}

mrw_result mrw_blob_skeleton_by_id(const mrw_blob *b, const mrw_id128 *id, mrw_skeleton_view *out) {
    if (!b || !id || !out) return MRW_E_RANGE;
    uint32_t st_off = b->section_table_off;
    for (uint32_t i = 0; i < b->section_count; ++i) {
        const uint8_t *e = section_entry(b->base, st_off, i);
        if (mrw_rd_u32(e) != MRW_SECTION_SKELETON) continue;
        const uint8_t *h = b->base + mrw_rd_u64(e + 8);
        mrw_id128 sid = mrw_rd_id128(h + 8);
        if (mrw_id_equal(&sid, id)) { fill_skeleton(h, out); return MRW_OK; }
    }
    return MRW_E_INCOMPATIBLE;
}

mrw_result mrw_blob_clip_by_id(const mrw_blob *b, const mrw_id128 *id, mrw_clip_view *out) {
    if (!b || !id || !out) return MRW_E_RANGE;
    uint32_t st_off = b->section_table_off;
    for (uint32_t i = 0; i < b->section_count; ++i) {
        const uint8_t *e = section_entry(b->base, st_off, i);
        if (mrw_rd_u32(e) != MRW_SECTION_CLIP) continue;
        const uint8_t *h = b->base + mrw_rd_u64(e + 8);
        mrw_id128 cid = mrw_rd_id128(h + 8);
        if (mrw_id_equal(&cid, id)) { fill_clip(h, out); return MRW_OK; }
    }
    return MRW_E_INCOMPATIBLE;
}

/* ------------------------------------------------------------------ typed array readers */

mrw_result mrw_skeleton_parent(const mrw_skeleton_view *v, uint32_t joint, uint16_t *out) {
    if (!v || !out) return MRW_E_RANGE;
    if (joint >= v->joint_count) return MRW_E_RANGE;
    *out = mrw_rd_u16(v->base + v->parent_off + (size_t)joint * MRW_PARENT_STRIDE);
    return MRW_OK;
}
mrw_result mrw_skeleton_rest_local(const mrw_skeleton_view *v, uint32_t joint, mrw_trs *out) {
    if (!v || !out) return MRW_E_RANGE;
    if (joint >= v->joint_count) return MRW_E_RANGE;
    memcpy(out, v->base + v->rest_local_off + (size_t)joint * MRW_REST_LOCAL_STRIDE, MRW_REST_LOCAL_STRIDE);
    return MRW_OK;
}
mrw_result mrw_skeleton_inverse_bind(const mrw_skeleton_view *v, uint32_t joint, float out_affine12[12]) {
    if (!v || !out_affine12) return MRW_E_RANGE;
    if (joint >= v->joint_count) return MRW_E_RANGE;
    memcpy(out_affine12, v->base + v->inverse_bind_off + (size_t)joint * MRW_INVERSE_BIND_STRIDE, MRW_INVERSE_BIND_STRIDE);
    return MRW_OK;
}
mrw_result mrw_skeleton_joint_name(const mrw_skeleton_view *v, uint32_t joint, const char **out_name) {
    if (!v || !out_name) return MRW_E_RANGE;
    if (joint >= v->joint_count) return MRW_E_RANGE;
    uint32_t no = mrw_rd_u32(v->base + v->name_off_off + (size_t)joint * MRW_NAME_OFF_STRIDE);
    *out_name = (const char *)(v->base + v->name_blob_off + no);
    return MRW_OK;
}
mrw_result mrw_clip_sample(const mrw_clip_view *v, uint32_t joint, uint32_t sample, mrw_trs *out) {
    if (!v || !out) return MRW_E_RANGE;
    if (joint >= v->joint_count || sample >= v->sample_count) return MRW_E_RANGE;
    uint64_t r = (uint64_t)joint * v->sample_count + sample;
    if (v->codec) {
        /* codec 1: on-disk sample is q4+t3 (28 B); map onto rot+trans, scale := (1,1,1). Mirrors
         * mrw_clip_root_sample - the runtime only consumes the codec, it never re-encodes. */
        memcpy(out, v->base + v->samples_off + r * MRW_ROOT_SAMPLE_STRIDE, MRW_ROOT_SAMPLE_STRIDE);
        out->scale[0] = out->scale[1] = out->scale[2] = 1.0f;
    } else {
        memcpy(out, v->base + v->samples_off + r * MRW_CLIP_SAMPLE_STRIDE, MRW_CLIP_SAMPLE_STRIDE);
    }
    return MRW_OK;
}
mrw_result mrw_clip_root_sample(const mrw_clip_view *v, uint32_t sample, mrw_xform *out) {
    if (!v || !out) return MRW_E_RANGE;
    if (!(v->flags & MRW_CLIP_HAS_ROOT_MOTION) || v->root_track_off == 0) return MRW_E_INCOMPATIBLE;
    if (sample >= v->sample_count) return MRW_E_RANGE;
    /* wire root sample is q4+t3 (28 B); map onto rot+trans, scale := 1 */
    memcpy(out, v->base + v->root_track_off + (size_t)sample * MRW_ROOT_SAMPLE_STRIDE, MRW_ROOT_SAMPLE_STRIDE);
    out->scale = 1.0f;
    return MRW_OK;
}
mrw_result mrw_baked_clip_entry(const mrw_baked_view *v, uint32_t clip_index, mrw_baked_clip *out) {
    if (!v || !out) return MRW_E_RANGE;
    if (clip_index >= v->clip_count) return MRW_E_RANGE;
    const uint8_t *ce = v->base + v->clip_table_off + (size_t)clip_index * MRW_BAKED_CLIP_STRIDE;
    out->clip_id          = mrw_rd_id128(ce + 0);
    out->first_frame      = mrw_rd_u32(ce + 16);
    out->frame_count      = mrw_rd_u32(ce + 20);
    out->source_duration_s= mrw_rd_f32(ce + 24);
    out->flags            = mrw_rd_u32(ce + 28);
    return MRW_OK;
}
