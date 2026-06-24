/* glTF marrow assets: load a tool-produced .mrw (SKELETON + CLIP[] + BAKED, made offline by
 * gltf2marrow -> marrow-bake) and read the matching skinned mesh straight from the source glTF with
 * cgltf. The result is a ProcAssets identical in shape to assets_proc_build's output - same
 * DemoVertex layout, same blob ownership - so crowd/heroes/validate consume it unchanged.
 *
 * The one integration hazard is the joint remap: glTF JOINTS_0 indices are in skin order, but the
 * .mrw skeleton is in marrow (DFS) order. We re-derive that order from the skin with the SAME helper
 * gltf2marrow used (tools/gltf2marrow/joint_order.h), giving a provably identical remap - no fragile
 * node-name matching (names are only a checked cross-check). */
#ifndef DEMO_ASSETS_GLTF_H
#define DEMO_ASSETS_GLTF_H

#include "assets_proc.h"   /* ProcAssets, DemoVertex, DemoClipInfo, assets_proc_free */

/* Load `mrw_path` (the pre-baked blob) + `gltf_path` (the source mesh/skin) into *out. The .mrw is
 * read into a 64-byte-aligned allocation kept alive in out->blob (mrw_blob_open precondition); the
 * whole bundle is released with assets_proc_free (shared with the procedural path). Returns 0 on
 * success (logs a one-line summary to stderr), nonzero on any error (logs the reason; *out is left
 * zeroed). */
int assets_gltf_build(const char *mrw_path, const char *gltf_path, ProcAssets *out);

#endif /* DEMO_ASSETS_GLTF_H */
