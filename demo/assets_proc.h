/* Procedural marrow assets: author a lean biped (skeleton + idle/walk/run clips) entirely in code
 * via the checked authoring library, bake every clip to a baked GPU palette stream with the
 * allocation-free bake core, and generate a matching skinned mesh (box segments per bone). No
 * external files - this is the self-contained content path. The marrow runtime only consumes the
 * resulting blob; authoring + baking live in tools/ (linked here, never in the runtime). */
#ifndef DEMO_ASSETS_PROC_H
#define DEMO_ASSETS_PROC_H

#include <stddef.h>
#include <stdint.h>

#define DEMO_PROC_MAX_CLIPS 4

/* One skinned-mesh vertex (bind space). Layout is mirrored by the GPU vertex input + a
 * _Static_assert in the renderer; bones are palette joint indices, weights pre-normalized (Σ=1). */
typedef struct {
    float    pos[3];
    float    nrm[3];
    float    tan[4];     /* xyz dir, w = bitangent handedness */
    uint32_t bones[4];
    float    weights[4];
} DemoVertex;

/* The graphics path reads this via explicit-offset vertex attributes, but the GPU validation path
 * (validate_skin.comp) reads it as a scalar-layout SSBO `Vtx` - so the C layout must match scalar
 * packing exactly (vec3s tightly packed, no 16-byte rounding) or the readback compares garbage. */
_Static_assert(sizeof(DemoVertex) == 72, "DemoVertex must be 72 bytes (matches scalar Vtx in validate_skin.comp)");
_Static_assert(offsetof(DemoVertex, pos)     ==  0, "DemoVertex.pos");
_Static_assert(offsetof(DemoVertex, nrm)     == 12, "DemoVertex.nrm");
_Static_assert(offsetof(DemoVertex, tan)     == 24, "DemoVertex.tan");
_Static_assert(offsetof(DemoVertex, bones)   == 40, "DemoVertex.bones");
_Static_assert(offsetof(DemoVertex, weights) == 56, "DemoVertex.weights");

typedef struct {
    const char *name;
    uint32_t    clip_index;     /* index into the blob's CLIP sections / baked clip table */
    float       duration_s;
    int         looping;
    int         has_root_motion;
} DemoClipInfo;

typedef struct {
    /* final .mrw blob: SKELETON + N CLIP + BAKED (free with assets_proc_free) */
    uint8_t  *blob;
    size_t    blob_size;
    uint32_t  joint_count;
    uint32_t  clip_count;
    DemoClipInfo clips[DEMO_PROC_MAX_CLIPS];

    /* procedural skinned mesh in bind space */
    DemoVertex *verts;   uint32_t vert_count;
    uint32_t   *indices; uint32_t index_count;

    int       tier_b_eligible;   /* did every clip pass the decomposability test */
    int       mesh_cull_back;    /* mesh winding is consistently CCW-outward -> back-face cullable
                                    (procedural box mesh: yes; arbitrary glTF: conservatively no) */
} ProcAssets;

/* Build everything. Returns 0 on success (logs a one-line summary + per-clip bake verdict to
 * stderr), nonzero on failure. */
int  assets_proc_build(ProcAssets *out);
void assets_proc_free(ProcAssets *out);

#endif /* DEMO_ASSETS_PROC_H */
