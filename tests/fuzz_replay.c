/* Portable corpus replayer for the .mrw loader fuzzer.
 *
 * Builds on ANY toolchain (no libFuzzer needed) and runs mrw_fuzz_one over each file passed on the
 * command line - so the committed corpus + crash reproducers are a permanent regression gate that
 * fires on every `ctest`, including the MSVC dev box where libFuzzer is unavailable. Under
 * MRW_SANITIZE it inherits ASan/UBSan, so a corpus replay traps real loader over-reads.
 *
 *   fuzz_replay <file>...                  replay each input; nonzero exit only on a crash
 *   fuzz_replay --check-coverage <file>... ALSO assert the corpus still covers every required shape
 *
 * The coverage check stops the "permanent regression gate" from silently rotting to an
 * empty/degenerate corpus: it fails unless the inputs that open MRW_OK collectively include every
 * shape the loader has distinct validation paths for. */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L   /* posix_memalign under strict -std=c11 */
#endif
#include "fuzz_blob.h"
#include "marrow.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <malloc.h>
static void *aligned64(size_t n) { return _aligned_malloc(n ? n : 1, 64); }
static void  aligned_free(void *p) { _aligned_free(p); }
#else
static void *aligned64(size_t n) { void *p = NULL; return posix_memalign(&p, 64, n ? n : 1) ? NULL : p; }
static void  aligned_free(void *p) { free(p); }
#endif

static uint8_t *read_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "fuzz_replay: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    uint8_t *buf = (uint8_t *)malloc((size_t)n ? (size_t)n : 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    *out_size = got;
    return buf;
}

typedef struct {
    int total_ok, skeleton_only, codec0, codec1, rootmotion, baked, baked_multiclip;
} cov;

/* Classify one input that opens cleanly into the corpus-coverage tally. */
static void classify(const uint8_t *data, size_t size, cov *c) {
    uint8_t *buf = (uint8_t *)aligned64(size);
    if (!buf) return;
    if (size) memcpy(buf, data, size);
    mrw_blob b;
    if (mrw_blob_open(buf, (uint64_t)size, &b) != MRW_OK) { aligned_free(buf); return; }

    c->total_ok++;
    int has_skel = 0, has_clip = 0, has_baked = 0;
    for (uint32_t i = 0; i < b.section_count; ++i) {
        uint32_t type = 0;
        mrw_blob_section_type(&b, i, &type);
        if (type == MRW_SECTION_SKELETON) {
            has_skel = 1;
        } else if (type == MRW_SECTION_CLIP) {
            has_clip = 1;
            mrw_clip_view cv;
            if (mrw_clip_view_at(&b, i, &cv) == MRW_OK) {
                if (cv.codec == 0) c->codec0 = 1; else if (cv.codec == 1) c->codec1 = 1;
                if (cv.flags & MRW_CLIP_HAS_ROOT_MOTION) c->rootmotion = 1;
            }
        } else if (type == MRW_SECTION_BAKED) {
            has_baked = 1;
            mrw_baked_view bv;
            if (mrw_baked_view_at(&b, i, &bv) == MRW_OK && bv.clip_count >= 2) c->baked_multiclip = 1;
        }
    }
    if (has_baked) c->baked = 1;
    if (has_skel && !has_clip && !has_baked) c->skeleton_only = 1;
    aligned_free(buf);
}

int main(int argc, char **argv) {
    int check = 0, start = 1;
    if (argc > 1 && strcmp(argv[1], "--check-coverage") == 0) { check = 1; start = 2; }
    if (argc <= start) {
        fprintf(stderr, "usage: fuzz_replay [--check-coverage] <file>...\n");
        return 2;
    }

    cov c = {0};
    for (int i = start; i < argc; ++i) {
        size_t size = 0;
        uint8_t *data = read_file(argv[i], &size);
        if (!data) return 2;
        mrw_fuzz_one(data, size);   /* the regression check: this must not crash on any input */
        if (check) classify(data, size, &c);
        free(data);
    }

    if (check) {
        int fail = 0;
        #define NEED(cond, what) do { if (!(cond)) { fprintf(stderr, \
            "fuzz_replay: corpus coverage gap - missing %s\n", what); fail = 1; } } while (0)
        NEED(c.total_ok > 0,       "any MRW_OK blob");
        NEED(c.skeleton_only,      "skeleton-only blob");
        NEED(c.codec0,             "codec-0 clip");
        NEED(c.codec1,             "codec-1 clip");
        NEED(c.rootmotion,         "root-motion clip");
        NEED(c.baked,              "baked section");
        NEED(c.baked_multiclip,    "multi-clip baked section");
        #undef NEED
        if (fail) return 1;
        printf("fuzz_replay: corpus coverage ok (%d OK blobs)\n", c.total_ok);
    } else {
        printf("fuzz_replay: replayed %d input(s), no crash\n", argc - start);
    }
    return 0;
}
