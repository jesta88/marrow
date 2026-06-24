/* Malformed-.mrw rejection. Build one valid skeleton+clip+baked blob,
 * then corrupt one field per case and assert mrw_blob_open returns the right code. */
#include "test_util.h"
#include "mrw_build.h"
#include "marrow_internal.h"   /* mrw_rd_*, wire offsets (test reuse) */
#include <string.h>

static const uint16_t PARENT[3] = { 0xFFFF, 0, 1 };
static const char *NAMES[3] = { "root", "j1", "j2" };
static const float REST[3*10] = {
    0,0,0,1, 0,0,0, 1,1,1,
    0,0,0,1, 1,0,0, 1,1,1,
    0,0,0,1, 1,0,0, 1,1,1,
};
static const float IB[3*12] = {
    1,0,0,0,  0,1,0,0, 0,0,1,0,
    1,0,0,-1, 0,1,0,0, 0,0,1,0,
    1,0,0,-2, 0,1,0,0, 0,0,1,0,
};
static const float SAMP[3*2*10] = {
    0,0,0,1,0,0,0,1,1,1,  0,0,0,1,0,0,0,1,1,1,
    0,0,0,1,1,0,0,1,1,1,  0,0,0,1,1,0,0,1,1,1,
    0,0,0,1,1,0,0,1,1,1,  0,0,0,1,1,0,0,1,1,1,
};

static uint8_t *G_BUF; static size_t G_SZ;

static void pk16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static void pk32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void pk64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }

#define TRY(label, expect, ...) do { \
    uint8_t *c = mrw_alloc64(G_SZ); memcpy(c, G_BUF, G_SZ); \
    { __VA_ARGS__; } \
    mrw_blob _b; mrw_result _rc = mrw_blob_open(c, G_SZ, &_b); \
    if (_rc != (expect)) { printf("FAIL %s: got %u expected %u\n", label, _rc, (unsigned)(expect)); ++g_fail; } \
    mrw_free(c); } while (0)

int main(void) {
    /* texels: every texel = (0,0,0,1) halves ⇒ unit quat, t=0, scale=1 */
    uint16_t tex[12*4];
    for (int i = 0; i < 12; ++i) { tex[i*4+0]=0; tex[i*4+1]=0; tex[i*4+2]=0; tex[i*4+3]=mrw_f32_to_half(1.0f); }
    uint32_t bidx[1] = {0}, bff[1] = {0}, bfc[1] = {2}, bflags[1] = {0};
    float bdur[1] = { 1.0f };
    mrw_baked baked = { 0, 2, tex, 1, bidx, bff, bfc, bdur, bflags };

    mrw_skel skel = { 3, PARENT, REST, IB, NAMES };
    mrw_clip clip = { 1.0f, 2, 0, SAMP, NULL };
    G_SZ = mrw_build(&skel, &clip, 1, &baked, &G_BUF);

    /* sanity: the pristine blob validates */
    mrw_blob ok; CHECK_EQ(mrw_blob_open(G_BUF, G_SZ, &ok), MRW_OK);
    CHECK_EQ(ok.section_count, 3);

    uint32_t st = mrw_rd_u32(G_BUF + 20);
    uint64_t skel_off = mrw_blob_section_offset(G_BUF, 0);
    uint64_t clip_off = mrw_blob_section_offset(G_BUF, 1);
    uint64_t baked_off = mrw_blob_section_offset(G_BUF, 2);
    uint32_t parent_off = mrw_rd_u32(G_BUF + skel_off + 24);
    uint32_t rest_off   = mrw_rd_u32(G_BUF + skel_off + 28);
    uint32_t bct_off    = mrw_rd_u32(G_BUF + baked_off + 40); /* clip_table_off */
    uint32_t tex_off    = mrw_rd_u32(G_BUF + baked_off + 44); /* texels_off     */

    /* ---- file header ---- */
    TRY("magic",        MRW_E_FORMAT, c[0] = 'X');
    TRY("version",      MRW_E_FORMAT, pk32(c + 4, 1));
    TRY("endian",       MRW_E_FORMAT, pk32(c + 8, 0x01020304u));
    TRY("file_flags",   MRW_E_FORMAT, pk32(c + 12, 1));
    TRY("blob_size",    MRW_E_FORMAT, pk64(c + 24, (uint64_t)G_SZ + 1));
    TRY("reserved",     MRW_E_FORMAT, c[40] = 1);

    /* ---- section table ---- */
    TRY("sec_off_align", MRW_E_ALIGN,  pk64(c + st + 32*1 + 8, clip_off + 1));
    TRY("sec_off_oob",   MRW_E_RANGE,  pk64(c + st + 32*1 + 8, 0x40000000u));
    TRY("sec_size_huge", MRW_E_RANGE,  pk64(c + st + 32*1 + 16, 0x100000000ull));
    TRY("sec_overlap",   MRW_E_FORMAT, pk64(c + st + 32*1 + 8, skel_off));
    TRY("sec_resv",      MRW_E_FORMAT, pk64(c + st + 32*0 + 24, 1));
    TRY("sec_flag_resv", MRW_E_FORMAT, pk32(c + st + 32*0 + 4, 2));
    TRY("type_unknown",  MRW_E_FORMAT, pk32(c + st + 32*2 + 0, 99)); /* baked->unknown, no OPTIONAL */
    TRY("type_optional", MRW_OK,       pk32(c + st + 32*2 + 0, 99); pk32(c + st + 32*2 + 4, MRW_SECTION_FLAG_OPTIONAL)); /* skip baked */

    /* ---- skeleton ---- */
    TRY("parent0",      MRW_E_FORMAT, pk16(c + skel_off + parent_off + 0, 0));
    TRY("parent_range", MRW_E_RANGE,  pk16(c + skel_off + parent_off + 2, 5));
    TRY("rest_nan",     MRW_E_RANGE,  pk32(c + skel_off + rest_off + 0, 0x7FC00000u));
    TRY("quat_nonunit", MRW_E_FORMAT, pk32(c + skel_off + rest_off + 0, 0x3F000000u)); /* qx=0.5 */

    /* ---- clip ---- */
    /* codec ∈ {0,1} are defined; ≥2 is reserved → MRW_E_UNSUPPORTED (blob stays well-formed). */
    TRY("clip_codec_unsup", MRW_E_UNSUPPORTED, pk32(c + clip_off + 4, 2));
    /* codec 1 reinterprets the samples at a 28-B (q4+t3) stride; on this codec-0 (40-B) blob that
     * lands a non-unit quat in the per-sample scan → MRW_E_FORMAT (codec-1 still validates its quats). */
    TRY("clip_codec1_nonunit", MRW_E_FORMAT, pk32(c + clip_off + 4, 1));
    TRY("clip_skel_id", MRW_E_INCOMPATIBLE, c[clip_off + 24] ^= 0xFF);

    /* ---- truncation: cut into the clip section ---- */
    do {
        size_t n = (size_t)clip_off + 8;
        uint8_t *c = mrw_alloc64(n); memcpy(c, G_BUF, n);
        pk64(c + 24, n);
        mrw_blob _b;
        CHECK_EQ(mrw_blob_open(c, n, &_b), MRW_E_RANGE);
        mrw_free(c);
    } while (0);

    /* ---- misaligned base ---- */
    do {
        uint8_t *c = mrw_alloc64(G_SZ + 64);
        memcpy(c + 1, G_BUF, G_SZ);
        mrw_blob _b;
        CHECK_EQ(mrw_blob_open(c + 1, G_SZ, &_b), MRW_E_ALIGN);
        mrw_free(c);
    } while (0);

    /* ---- baked ---- */
    TRY("baked_encoding",   MRW_E_UNSUPPORTED,  pk32(c + baked_off + 0, 2));
    TRY("baked_tpb",        MRW_E_FORMAT,       pk32(c + baked_off + 8, 3));
    TRY("baked_stride",     MRW_E_FORMAT,       pk32(c + baked_off + 12, 5)); /* < bone_count*2=6 */
    TRY("baked_texelcount", MRW_E_FORMAT,       pk32(c + baked_off + 48, 13));
    TRY("baked_loop_mism",  MRW_E_FORMAT,       pk32(c + baked_off + bct_off + 28, MRW_BAKED_CLIP_LOOPING));
    TRY("baked_clip_id",    MRW_E_INCOMPATIBLE, c[baked_off + bct_off + 0] ^= 0xFF);
    TRY("baked_dur",        MRW_E_FORMAT,       pk32(c + baked_off + bct_off + 24, 0x40A00000u)); /* 5.0f != 1.0f */
    /* one-frame baked cache (dur 0) pointing at a dynamic clip (dur 1.0) ⇒ rejected */
    TRY("baked_static_dynamic", MRW_E_FORMAT,
        pk32(c + baked_off + bct_off + 20, 1);                /* frame_count = 1 */
        pk32(c + baked_off + bct_off + 24, 0));               /* source_duration = 0 */
    /* clip frame range out of the baked stream: first_frame+frame_count > total_frames(=2) */
    TRY("baked_framerange",  MRW_E_RANGE,  pk32(c + baked_off + bct_off + 20, 3)); /* frame_count = 3 */
    /* non-finite texel (a quat half = +inf) */
    TRY("baked_texel_inf",   MRW_E_RANGE,  pk16(c + baked_off + tex_off + 0, 0x7C00u));
    /* degenerate baked quaternion (finite but far from unit: qx = 0.5 ⇒ ‖q‖²=1.25) */
    TRY("baked_quat_degen",  MRW_E_FORMAT, pk16(c + baked_off + tex_off + 0, mrw_f32_to_half(0.5f)));

    mrw_free(G_BUF);
    printf(g_fail ? "test_malformed: %d FAILED\n" : "test_malformed: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
