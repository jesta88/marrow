/* marrow-bake CLI - bake a v0 .mrw clip set into a baked GPU crowd asset (BAKED section),
 * iff the rig passes the decomposability test; otherwise emit a clip-set-only copy. Separate
 * target from the zero-dependency runtime; the bake lives in bake_run.c. */
#include "bake_run.h"
#include "mrw_authoring.h"   /* mrw_authoring_free */

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Checked numeric parse - reject trailing junk (so "--mesh-skin abc" can't silently become 0). */
static int parse_float(const char *s, float *out) {
    char *e; double v = strtod(s, &e);
    if (e == s || *e != '\0') return -1;
    *out = (float)v; return 0;
}
static int parse_int(const char *s, int *out) {
    char *e; long v = strtol(s, &e, 10);
    if (e == s || *e != '\0' || v < INT_MIN || v > INT_MAX) return -1;
    *out = (int)v; return 0;
}

static int usage(FILE *f, int code) {
    fprintf(f,
        "marrow-bake - bake a v0 .mrw clip set into a Tier-B crowd asset.\n"
        "\n"
        "usage: marrow-bake <rig.mrw> -o <out.mrw> [options]\n"
        "\n"
        "options:\n"
        "  -o <path>                 output .mrw file (required)\n"
        "  --bake-fps <n>            baked frame rate, default 30 (a dynamic clip bakes >= 2 frames;\n"
        "                            the source clip duration is preserved in the clip table)\n"
        "  --decompose-tolerance <m> residual tolerance in metres; default max(1 mm, 0.2%% of the\n"
        "                            model-AABB diagonal)\n"
        "  --probe-radius <m>        no-mesh / fallback probe box half-extent, default 0.05\n"
        "  --mesh <file.gltf|.glb>   derive probes from the mesh's actual skin influences (per bone)\n"
        "  --mesh-skin <i>           select skin index i in --mesh (default: require exactly one skin)\n"
        "  --allow-probe-fallback    with --mesh: box-fallback unmapped joints instead of erroring\n"
        "  --require-baked           exit nonzero if the rig is ineligible for Tier B (still writes the\n"
        "                            Tier-A asset)\n"
        "  -h, --help                show this help\n"
        "\n"
        "notes:\n"
        "  Input is a Tier-A clip set (one SKELETON + >=1 CLIP), e.g. gltf2marrow output. An ineligible\n"
        "  rig stays Tier A: the output is a valid .mrw without a BAKED section (no full-matrix fallback).\n");
    return code;
}

int main(int argc, char **argv) {
    const char *input = NULL, *output = NULL;
    mrw_bake_options opt = {0};
    opt.bake_fps = 30.0f;
    opt.mesh_skin_index = -1;
    int require_baked = 0;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) return usage(stdout, 0);
        else if (strcmp(a, "-o") == 0) { if (++i >= argc) { fprintf(stderr, "marrow-bake: -o needs a path\n"); return 2; } output = argv[i]; }
        else if (strcmp(a, "--bake-fps") == 0) { if (++i >= argc) { fprintf(stderr, "marrow-bake: --bake-fps needs a value\n"); return 2; } if (parse_float(argv[i], &opt.bake_fps)) { fprintf(stderr, "marrow-bake: --bake-fps: not a number: '%s'\n", argv[i]); return 2; } }
        else if (strcmp(a, "--decompose-tolerance") == 0) { if (++i >= argc) { fprintf(stderr, "marrow-bake: --decompose-tolerance needs a value\n"); return 2; } if (parse_float(argv[i], &opt.decompose_tol)) { fprintf(stderr, "marrow-bake: --decompose-tolerance: not a number: '%s'\n", argv[i]); return 2; } }
        else if (strcmp(a, "--probe-radius") == 0) { if (++i >= argc) { fprintf(stderr, "marrow-bake: --probe-radius needs a value\n"); return 2; } if (parse_float(argv[i], &opt.probe_radius)) { fprintf(stderr, "marrow-bake: --probe-radius: not a number: '%s'\n", argv[i]); return 2; } }
        else if (strcmp(a, "--mesh") == 0) { if (++i >= argc) { fprintf(stderr, "marrow-bake: --mesh needs a path\n"); return 2; } opt.mesh_path = argv[i]; }
        else if (strcmp(a, "--mesh-skin") == 0) { if (++i >= argc) { fprintf(stderr, "marrow-bake: --mesh-skin needs an index\n"); return 2; } if (parse_int(argv[i], &opt.mesh_skin_index)) { fprintf(stderr, "marrow-bake: --mesh-skin: not an integer: '%s'\n", argv[i]); return 2; } }
        else if (strcmp(a, "--allow-probe-fallback") == 0) { opt.allow_probe_fallback = 1; }
        else if (strcmp(a, "--require-baked") == 0) { require_baked = 1; }
        else if (a[0] == '-' && a[1] != '\0') { fprintf(stderr, "marrow-bake: unknown option '%s'\n", a); return usage(stderr, 2); }
        else if (!input) { input = a; }
        else { fprintf(stderr, "marrow-bake: unexpected argument '%s'\n", a); return usage(stderr, 2); }
    }

    if (!input)  { fprintf(stderr, "marrow-bake: missing input .mrw\n"); return usage(stderr, 2); }
    if (!output) { fprintf(stderr, "marrow-bake: missing -o <out.mrw>\n"); return usage(stderr, 2); }
    if (!(opt.bake_fps > 0.0f) || !isfinite(opt.bake_fps)) { fprintf(stderr, "marrow-bake: --bake-fps must be a finite value > 0\n"); return 2; }
    opt.input_path = input;

    uint8_t *buf = NULL; size_t size = 0;
    int eligible = 0; mrw_bake_stats worst; char diag[256] = {0};
    mrw_result r = mrw_bake_run(&opt, &buf, &size, &eligible, &worst, diag, sizeof diag);
    if (r != MRW_OK) {
        fprintf(stderr, "marrow-bake: %s\n", diag[0] ? diag : "bake failed");
        return 1;
    }

    FILE *f = fopen(output, "wb");
    if (!f) { fprintf(stderr, "marrow-bake: cannot open '%s' for writing\n", output); mrw_authoring_free(buf); return 1; }
    size_t wrote = fwrite(buf, 1, size, f);
    int ferr = (wrote != size);
    if (fclose(f) != 0) ferr = 1;
    mrw_authoring_free(buf);
    if (ferr) { fprintf(stderr, "marrow-bake: write error on '%s'\n", output); return 1; }

    /* diag carries the one-line summary in both the eligible and ineligible cases. */
    fprintf(stderr, "marrow-bake: %s -> %s (%zu bytes)\n", diag, output, size);
    if (!eligible && require_baked) {
        fprintf(stderr, "marrow-bake: --require-baked set and rig is ineligible for Tier B\n");
        return 1;
    }
    return 0;
}
