/*
 * marrow quickstart — load a .mrw, animate one character, print its skinning palette.
 *
 * This is the end-to-end Tier-A path in ~60 lines: open and validate a blob, take the
 * skeleton view, find a clip, and produce the canonical 3x4 skinning palette at a given
 * time. It is built and run by CTest against fixtures/rig.mrw, so the snippet quoted in
 * the documentation cannot drift from code that compiles and runs.
 *
 *   mrw_quickstart [path/to/file.mrw]      (defaults to fixtures/rig.mrw)
 */
#include "marrow.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* marrow itself never allocates — but YOUR app must hand mrw_blob_open a >=64-byte-aligned
 * buffer. aligned_alloc is C11; MSVC spells it _aligned_malloc. */
#if defined(_MSC_VER)
#  include <malloc.h>
static void *aligned_block(size_t align, size_t size) { return _aligned_malloc(size, align); }
static void  aligned_block_free(void *p) { _aligned_free(p); }
#else
static void *aligned_block(size_t align, size_t size) {
    /* C11 aligned_alloc requires size to be a multiple of the alignment. */
    size_t rem = size % align;
    if (rem) size += align - rem;
    return aligned_alloc(align, size);
}
static void  aligned_block_free(void *p) { free(p); }
#endif

/* Read an entire file into a freshly allocated 64-byte-aligned buffer. */
static uint8_t *read_aligned(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return NULL; }
    uint8_t *buf = (uint8_t *)aligned_block(64, (size_t)n);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { aligned_block_free(buf); return NULL; }
    *out_size = (size_t)n;
    return buf;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "fixtures/rig.mrw";

    size_t size = 0;
    uint8_t *data = read_aligned(path, &size);
    if (!data) { fprintf(stderr, "cannot read %s\n", path); return 1; }

    /* Validate, then view. After MRW_OK the blob is safe to read through the accessors. */
    mrw_blob blob;
    if (mrw_blob_open(data, size, &blob) != MRW_OK) {
        fprintf(stderr, "%s is not a valid .mrw file\n", path);
        aligned_block_free(data);
        return 1;
    }

    /* A .mrw has at most one skeleton; grab its view. */
    mrw_skeleton_view skel;
    if (mrw_blob_skeleton(&blob, &skel) != MRW_OK) {
        fprintf(stderr, "no skeleton in %s\n", path);
        aligned_block_free(data);
        return 1;
    }
    printf("skeleton: %u joints\n", skel.joint_count);

    /* Find the first clip by scanning section types — never assume a fixed section index. */
    mrw_clip_view clip;
    int have_clip = 0;
    for (uint32_t i = 0; i < blob.section_count; ++i) {
        uint32_t type = 0;
        if (mrw_blob_section_type(&blob, i, &type) == MRW_OK && type == MRW_SECTION_CLIP) {
            if (mrw_clip_view_at(&blob, i, &clip) == MRW_OK) { have_clip = 1; break; }
        }
    }
    if (!have_clip) { fprintf(stderr, "no clip in %s\n", path); aligned_block_free(data); return 1; }

    /* Sample halfway through the clip. A dense clip's duration is (sample_count-1)/fps. */
    float duration = clip.sample_count > 1 ? (float)(clip.sample_count - 1) / clip.fps : 0.0f;
    float t = duration * 0.5f;

    /* Caller-owned, >=16-byte-aligned scratch + output: joint_count * 12 floats each. */
    uint32_t jc = skel.joint_count;
    float *scratch = (float *)aligned_block(16, (size_t)jc * 12 * sizeof(float));
    float *palette = (float *)aligned_block(16, (size_t)jc * 12 * sizeof(float));
    if (!scratch || !palette) {
        fprintf(stderr, "out of memory\n");
        aligned_block_free(scratch); aligned_block_free(palette); aligned_block_free(data);
        return 1;
    }

    /* Fused: sample @ t -> hierarchy compose -> apply inverse-bind. */
    mrw_result r = mrw_clip_to_palette(&skel, &clip, t, scratch, palette, jc);
    if (r != MRW_OK) {
        fprintf(stderr, "mrw_clip_to_palette failed: %u\n", r);
        aligned_block_free(scratch); aligned_block_free(palette); aligned_block_free(data);
        return 1;
    }

    /* palette[0..12) is joint 0's 3x4 skinning matrix (rows row0, row1, row2). */
    printf("clip: %u samples @ %.1f fps (%.3fs); sampled t=%.3f\n",
           clip.sample_count, (double)clip.fps, (double)duration, (double)t);
    printf("joint 0 palette row0 = [% .3f % .3f % .3f | % .3f]\n",
           (double)palette[0], (double)palette[1], (double)palette[2], (double)palette[3]);

    aligned_block_free(scratch);
    aligned_block_free(palette);
    aligned_block_free(data);
    return 0;
}
