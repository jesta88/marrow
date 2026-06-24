/* gltf2marrow conversion core: parse a glTF 2.0 file, fold its node hierarchy onto a marrow joint
 * skeleton, resample its animations onto dense fixed-rate codec-0 clips, and serialize a v0 .mrw
 * blob. Library half of the tool (no main); main.c wraps it in a CLI. */
#ifndef MRW_G2M_CONVERT_H
#define MRW_G2M_CONVERT_H

#include "marrow.h"

typedef struct {
    const char *input_path;     /* .gltf or .glb                                       */
    float       fps;            /* nominal resample rate (default 30)                  */
    int32_t     skin_index;     /* -1 = require exactly one skin; else select that skin */
    int         loop;           /* set LOOPING on emitted clips (warns on non-closure) */
    int         force_codec0;   /* skip the unit-scale codec-1 snap; always emit raw codec 0 */
    const char *const *anims;   /* NULL = all animations; else only those named here   */
    uint32_t    anim_count;     /* length of anims[]                                   */
} mrw_g2m_options;

/* Parse + fold/resample + serialize into a fresh 64-aligned blob (free with mrw_authoring_free).
 * On MRW_OK fills *out_buf/*out_size; on any error returns a code and writes a human-readable
 * reason into diag (if diag != NULL && diag_cap > 0). Non-fatal notes (ignored weight channels,
 * nominal-vs-stored fps, loop-seam gaps) are logged to stderr. */
mrw_result mrw_g2m_convert(const mrw_g2m_options *opt, uint8_t **out_buf, size_t *out_size,
                           char *diag, size_t diag_cap);

#endif /* MRW_G2M_CONVERT_H */
