/* gltf2marrow CLI - import a glTF 2.0 skeleton + animations and emit a v0 .mrw blob.
 * Separate target from the zero-dependency runtime; the conversion lives in convert.c. */
#include "convert.h"
#include "mrw_authoring.h"   /* mrw_authoring_free */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ANIMS 256

static int usage(FILE *f, int code) {
    fprintf(f,
        "gltf2marrow - import a glTF 2.0 skeleton + animations into a v0 .mrw blob.\n"
        "\n"
        "usage: gltf2marrow <input.gltf|.glb> -o <out.mrw> [options]\n"
        "\n"
        "options:\n"
        "  -o <path>       output .mrw file (required)\n"
        "  --fps <n>       nominal resample rate, default 30 (clip duration is preserved exactly;\n"
        "                  the stored fps is back-solved so (sample_count-1)/fps == duration)\n"
        "  --skin <i>      select skin index i (default: require exactly one skin)\n"
        "  --loop          mark emitted clips LOOPING (warns if a clip does not close)\n"
        "  --codec0        force raw codec 0 (skip the unit-scale codec-1 snap; for the codec sweep)\n"
        "  --anim <name>   convert only the named animation (repeatable; default: all)\n"
        "  -h, --help      show this help\n"
        "\n"
        "notes:\n"
        "  glTF and marrow share conventions (right-handed, +Y up, m/s/rad), so no axis conversion.\n"
        "  Root-motion extraction and Tier-B baking are out of scope (see marrow-bake).\n");
    return code;
}

int main(int argc, char **argv) {
    const char *input = NULL, *output = NULL;
    const char *anims[MAX_ANIMS]; uint32_t anim_count = 0;
    mrw_g2m_options opt = {0};
    opt.fps = 30.0f;
    opt.skin_index = -1;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) return usage(stdout, 0);
        else if (strcmp(a, "-o") == 0) { if (++i >= argc) { fprintf(stderr, "gltf2marrow: -o needs a path\n"); return 2; } output = argv[i]; }
        else if (strcmp(a, "--fps") == 0) { if (++i >= argc) { fprintf(stderr, "gltf2marrow: --fps needs a value\n"); return 2; } opt.fps = (float)atof(argv[i]); }
        else if (strcmp(a, "--skin") == 0) { if (++i >= argc) { fprintf(stderr, "gltf2marrow: --skin needs an index\n"); return 2; } opt.skin_index = atoi(argv[i]); }
        else if (strcmp(a, "--loop") == 0) { opt.loop = 1; }
        else if (strcmp(a, "--codec0") == 0) { opt.force_codec0 = 1; }
        else if (strcmp(a, "--anim") == 0) {
            if (++i >= argc) { fprintf(stderr, "gltf2marrow: --anim needs a name\n"); return 2; }
            if (anim_count >= MAX_ANIMS) { fprintf(stderr, "gltf2marrow: too many --anim (max %d)\n", MAX_ANIMS); return 2; }
            anims[anim_count++] = argv[i];
        }
        else if (a[0] == '-' && a[1] != '\0') { fprintf(stderr, "gltf2marrow: unknown option '%s'\n", a); return usage(stderr, 2); }
        else if (!input) { input = a; }
        else { fprintf(stderr, "gltf2marrow: unexpected argument '%s'\n", a); return usage(stderr, 2); }
    }

    if (!input)  { fprintf(stderr, "gltf2marrow: missing input file\n"); return usage(stderr, 2); }
    if (!output) { fprintf(stderr, "gltf2marrow: missing -o <out.mrw>\n"); return usage(stderr, 2); }

    opt.input_path = input;
    opt.anims = anim_count ? anims : NULL;
    opt.anim_count = anim_count;
    if (!(opt.fps > 0.0f) || !isfinite(opt.fps)) { fprintf(stderr, "gltf2marrow: --fps must be a finite value > 0\n"); return 2; }

    uint8_t *buf = NULL; size_t size = 0;
    char diag[256] = {0};
    mrw_result r = mrw_g2m_convert(&opt, &buf, &size, diag, sizeof diag);
    if (r != MRW_OK) {
        fprintf(stderr, "gltf2marrow: conversion failed: %s\n", diag[0] ? diag : "(no detail)");
        return 1;
    }

    FILE *f = fopen(output, "wb");
    if (!f) { fprintf(stderr, "gltf2marrow: cannot open '%s' for writing\n", output); mrw_authoring_free(buf); return 1; }
    size_t wrote = fwrite(buf, 1, size, f);
    int ferr = (wrote != size);
    if (fclose(f) != 0) ferr = 1;
    mrw_authoring_free(buf);
    if (ferr) { fprintf(stderr, "gltf2marrow: write error on '%s'\n", output); return 1; }

    fprintf(stderr, "gltf2marrow: wrote %s (%zu bytes)\n", output, size);
    return 0;
}
