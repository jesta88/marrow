/* marrow-bake front-end: read a v0 .mrw clip set, bake every clip to a baked GPU palette stream via
 * the bake core, and re-serialize through the checked authoring writer - producing a baked crowd
 * asset when the rig passes the decomposability test, or a clip-set-only copy when it does not.
 * Separate target from the zero-dependency runtime; uses cgltf only for the optional --mesh probe
 * path. The pure bake math lives in mrw_bake.{h,c}; this file owns I/O, struct reconstruction, probe
 * geometry, frame-count/duration policy, assembly, and self-validation. */
#ifndef MRW_BAKE_RUN_H
#define MRW_BAKE_RUN_H

#include "marrow.h"
#include "mrw_bake.h"   /* mrw_bake_stats */

typedef struct {
    const char *input_path;           /* v0 .mrw clip set (one SKELETON + ≥1 CLIP); required        */
    const char *mesh_path;            /* optional .glb/.gltf for mesh-derived probes; NULL ⇒ AABB box*/
    int32_t     mesh_skin_index;      /* -1 = require exactly one skin in the mesh glTF              */
    int         allow_probe_fallback; /* with --mesh: tolerate unmapped joints (box fallback + warn) */
    float       bake_fps;             /* baked frame rate (default 30; finite > 0)                   */
    float       decompose_tol;        /* metres; ≤0 ⇒ default max(1 mm, 0.2%·model-AABB diagonal)    */
    float       probe_radius;         /* no-mesh / fallback box half-extent in metres (default 0.05) */
} mrw_bake_options;

/* Read + bake + serialize into a fresh 64-aligned blob (free with mrw_authoring_free). On MRW_OK,
 * *out_buf/*out_size ALWAYS hold a valid, self-validated blob: baked when *out_eligible == 1, else a
 * clip-set-only copy of the input (skeleton + clips, no BAKED). *out_worst gets the worst stats across
 * the clip set. Ineligibility is a NORMAL MRW_OK outcome (the CLI decides the exit code via
 * --require-baked). `diag` (if non-NULL, cap > 0) always gets a human-readable one-line summary:
 * the bake result on success/ineligible, or the failure reason on error. On any MRW_E_* nothing is
 * allocated and *out_buf is left NULL. */
mrw_result mrw_bake_run(const mrw_bake_options *opt, uint8_t **out_buf, size_t *out_size,
                        int *out_eligible, mrw_bake_stats *out_worst, char *diag, size_t diag_cap);

#endif /* MRW_BAKE_RUN_H */
