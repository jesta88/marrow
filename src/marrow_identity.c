/* marrow content identities - FNV-1a-128 over a layout-independent canonical stream.
 * Multi-byte values are little-endian; the section's own id fields and any embedded
 * skeleton_id are excluded. The runtime treats ids as opaque (equality only); these
 * functions exist for tools/tests and to freeze the golden vectors. */
#include "marrow_internal.h"

/* FNV-1a-128 parameters, as 64-bit hi/lo limbs:
 *   offset_basis = 0x6c62272e07bb014262b821756295c58d
 *   prime        = 0x0000000001000000000000000000013B  (= 2^88 + 0x13B) */
#define FNV_BASIS_HI 0x6c62272e07bb0142ull
#define FNV_BASIS_LO 0x62b821756295c58dull
#define FNV_PRIME_HI 0x0000000001000000ull
#define FNV_PRIME_LO 0x000000000000013Bull

/* full 64×64 → 128 multiply, portable (no __int128 / intrinsics) */
static void mul64(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo) {
    uint64_t a_lo = (uint32_t)a, a_hi = a >> 32;
    uint64_t b_lo = (uint32_t)b, b_hi = b >> 32;
    uint64_t ll = a_lo * b_lo;
    uint64_t lh = a_lo * b_hi;
    uint64_t hl = a_hi * b_lo;
    uint64_t hh = a_hi * b_hi;
    uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFull) + (hl & 0xFFFFFFFFull);
    *lo = (ll & 0xFFFFFFFFull) | (mid << 32);
    *hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}

typedef struct { uint64_t hi, lo; } fnv_state;

static void fnv_init(fnv_state *s) { s->hi = FNV_BASIS_HI; s->lo = FNV_BASIS_LO; }

static void fnv_byte(fnv_state *s, uint8_t x) {
    s->lo ^= (uint64_t)x;                         /* XOR into the low byte */
    /* h *= prime  (mod 2^128); the hi·hi term shifts out of 128 bits */
    uint64_t ll_hi, ll_lo;
    mul64(s->lo, FNV_PRIME_LO, &ll_hi, &ll_lo);
    uint64_t r_hi = ll_hi + s->lo * FNV_PRIME_HI + s->hi * FNV_PRIME_LO;
    s->lo = ll_lo;
    s->hi = r_hi;
}

static void fnv_bytes(fnv_state *s, const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; ++i) fnv_byte(s, p[i]);
}
static void fnv_u32(fnv_state *s, uint32_t v) { uint8_t b[4]; memcpy(b, &v, 4); fnv_bytes(s, b, 4); }
static void fnv_f32(fnv_state *s, float v)    { uint8_t b[4]; memcpy(b, &v, 4); fnv_bytes(s, b, 4); }

static void fnv_finish(const fnv_state *s, mrw_id128 *out) { out->lo = s->lo; out->hi = s->hi; }

int mrw_id_equal(const mrw_id128 *a, const mrw_id128 *b) {
    return a->lo == b->lo && a->hi == b->hi;
}

/* ---- canonical streams (byte sources; reads are strict-aliasing-safe via memcpy) ---- */

static mrw_result identity_skeleton_core(
    uint32_t joint_count, uint32_t name_blob_size,
    const uint8_t *parent, const uint8_t *rest_local, const uint8_t *inverse_bind,
    const uint8_t *name_off, const uint8_t *name_blob, mrw_id128 *out)
{
    if (!out) return MRW_E_RANGE;
    /* Bound joint_count to the wire limit so the contiguous-block byte counts below
     * cannot overflow size_t and hash a wrapped length (the loader already enforces this for
     * views; this guards the raw public API against untrusted tool inputs). */
    if (joint_count < 1 || joint_count > 0xFFFEu) return MRW_E_RANGE;
    fnv_state s; fnv_init(&s);
    fnv_byte(&s, 0x01);
    fnv_u32(&s, joint_count);
    /* parents (u16 LE), rest_local (10 f32), inverse_bind (12 f32): contiguous blocks */
    fnv_bytes(&s, parent,       (size_t)joint_count * MRW_PARENT_STRIDE);
    fnv_bytes(&s, rest_local,   (size_t)joint_count * MRW_REST_LOCAL_STRIDE);
    fnv_bytes(&s, inverse_bind, (size_t)joint_count * MRW_INVERSE_BIND_STRIDE);
    for (uint32_t j = 0; j < joint_count; ++j) {
        uint32_t off = mrw_rd_u32(name_off + (size_t)j * MRW_NAME_OFF_STRIDE);
        const uint8_t *nm = name_blob + off;
        uint32_t len = 0;
        while (off + len < name_blob_size && nm[len] != 0) ++len; /* length sans NUL */
        fnv_u32(&s, len);
        fnv_bytes(&s, nm, len);
    }
    fnv_finish(&s, out);
    return MRW_OK;
}

static mrw_result identity_clip_core(
    uint32_t joint_count, float fps, uint32_t sample_count, uint32_t flags, uint32_t codec,
    const uint8_t *samples, const uint8_t *root_track, mrw_id128 *out)
{
    if (!out) return MRW_E_RANGE;
    if (joint_count < 1 || joint_count > 0xFFFEu) return MRW_E_RANGE;
    /* overflow-safe byte counts (untrusted raw-API inputs): joint<=0xFFFE bounds the
     * product, and we still reject if it cannot be represented as size_t. The CANONICAL stream is
     * always 10 f32/sample (q4+t3+s3) regardless of codec, so this 40-byte bound also covers the
     * codec-1 per-sample read (28 on-disk B) plus the three appended 1.0f. */
    uint64_t nsamp = (uint64_t)joint_count * sample_count;
    uint64_t samp_bytes = nsamp * MRW_CLIP_SAMPLE_STRIDE;
    uint64_t root_bytes = (uint64_t)sample_count * MRW_ROOT_SAMPLE_STRIDE;
    if (samp_bytes > (uint64_t)SIZE_MAX || root_bytes > (uint64_t)SIZE_MAX) return MRW_E_OVERFLOW;

    uint32_t mask = flags & (MRW_CLIP_LOOPING | MRW_CLIP_HAS_ROOT_MOTION);
    fnv_state s; fnv_init(&s);
    fnv_byte(&s, 0x02);                  /* clip-stream tag - codec-INDEPENDENT (codec is not hashed) */
    fnv_u32(&s, joint_count);
    fnv_f32(&s, fps);
    fnv_u32(&s, sample_count);
    fnv_u32(&s, mask);
    if (codec == 0) {
        /* codec 0: on-disk bytes already equal the canonical 10-f32/sample stream - one block. */
        fnv_bytes(&s, samples, (size_t)samp_bytes);
    } else {
        /* codec 1: the canonical stream is q4+t3+(1,1,1). Stream it WITHOUT materializing a temp
         * 10-float sample: per sample, hash the 28 on-disk bytes (q4+t3) then three little-endian
         * 1.0f. So id(codec1) == id(codec0) iff the codec-0 scales are bit-exactly 1.0f. */
        for (uint64_t r = 0; r < nsamp; ++r) {
            fnv_bytes(&s, samples + (size_t)r * MRW_ROOT_SAMPLE_STRIDE, MRW_ROOT_SAMPLE_STRIDE); /* 28 */
            fnv_f32(&s, 1.0f); fnv_f32(&s, 1.0f); fnv_f32(&s, 1.0f);
        }
    }
    if (flags & MRW_CLIP_HAS_ROOT_MOTION)
        fnv_bytes(&s, root_track, (size_t)root_bytes);
    fnv_finish(&s, out);
    return MRW_OK;
}

/* ---- public raw API (native arrays) ---- */

mrw_result mrw_skeleton_compute_id(
    uint32_t joint_count, uint32_t name_blob_size,
    const uint16_t *parent, const float *rest_local, const float *inverse_bind,
    const uint32_t *name_off, const uint8_t *name_blob, mrw_id128 *out)
{
    return identity_skeleton_core(joint_count, name_blob_size,
        (const uint8_t *)parent, (const uint8_t *)rest_local, (const uint8_t *)inverse_bind,
        (const uint8_t *)name_off, name_blob, out);
}

mrw_result mrw_clip_compute_id(
    uint32_t joint_count, float fps, uint32_t sample_count, uint32_t flags,
    const float *samples, const float *root_track, mrw_id128 *out)
{
    const uint8_t *root = (flags & MRW_CLIP_HAS_ROOT_MOTION) ? (const uint8_t *)root_track : NULL;
    /* Raw native-array API: `samples` is the canonical 10-f32/sample stream (codec 0). A codec-1 clip
     * id equals this over a scale-snapped-to-exactly-1.0f buffer (see identity_clip_core). */
    return identity_clip_core(joint_count, fps, sample_count, flags, 0u, (const uint8_t *)samples, root, out);
}

/* ---- view wrappers (blob bytes) ---- */

mrw_result mrw_skeleton_view_id(const mrw_skeleton_view *v, mrw_id128 *out) {
    return identity_skeleton_core(v->joint_count, v->name_blob_size,
        v->base + v->parent_off, v->base + v->rest_local_off, v->base + v->inverse_bind_off,
        v->base + v->name_off_off, v->base + v->name_blob_off, out);
}

mrw_result mrw_clip_view_id(const mrw_clip_view *v, mrw_id128 *out) {
    const uint8_t *root = (v->flags & MRW_CLIP_HAS_ROOT_MOTION) ? v->base + v->root_track_off : NULL;
    return identity_clip_core(v->joint_count, v->fps, v->sample_count, v->flags, v->codec,
        v->base + v->samples_off, root, out);
}
