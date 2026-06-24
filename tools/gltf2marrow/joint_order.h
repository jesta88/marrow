/* Canonical marrow joint ordering for a glTF skin - the single definition of "which marrow joint
 * index a skin-order joint maps to". gltf2marrow serializes the skeleton/clips in this order; any
 * consumer that must address that skeleton from glTF data (notably the demo's mesh loader, whose
 * JOINTS_0 indices are in skin order) remaps through `marrow_of`. Sharing one definition lets the
 * demo derive a provably identical remap by running the same DFS - no fragile node-name matching.
 *
 * Resolution rules (single-tree skeleton): each joint's parent is its nearest ancestor that is
 * itself a skin joint; exactly one joint must be parentless (the root); the marrow order is a DFS
 * from the root with children visited in ascending skin-order, parent always before child. */
#ifndef MRW_G2M_JOINT_ORDER_H
#define MRW_G2M_JOINT_ORDER_H

#include "marrow.h"   /* mrw_result */
#include "cgltf.h"    /* cgltf_skin */

typedef struct {
    uint32_t  joint_count;   /* == skin->joints_count                                     */
    int32_t  *parent_js;     /* [joint_count] skin-order joint -> parent skin-order (-1 root) */
    uint32_t *order;         /* [joint_count] marrow index -> skin-order joint index       */
    uint32_t *marrow_of;     /* [joint_count] skin-order joint -> marrow index             */
    int32_t   root_js;       /* skin-order index of the single root                        */
} mrw_g2m_joint_order;

/* Build the ordering for `skin`. On MRW_OK fills *out (free with mrw_g2m_joint_order_free). On any
 * error returns a code, writes a human-readable reason into diag (when diag != NULL && diag_cap > 0),
 * and leaves *out zeroed. Errors: MRW_E_RANGE (joint count not in [1, 0xFFFE]), MRW_E_FORMAT (not a
 * single-rooted tree - zero/multiple roots, or a cycle), MRW_E_OVERFLOW (allocation failed). */
mrw_result mrw_g2m_joint_order_build(const cgltf_skin *skin, mrw_g2m_joint_order *out,
                                     char *diag, size_t diag_cap);

void mrw_g2m_joint_order_free(mrw_g2m_joint_order *jo);

#endif /* MRW_G2M_JOINT_ORDER_H */
