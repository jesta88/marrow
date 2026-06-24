/* See joint_order.h. The converter and any external consumer (the demo mesh loader) produce a
 * bit-identical skin_joint -> marrow_index remap by running this same DFS. */
#include "joint_order.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_diag(char *diag, size_t cap, const char *fmt, ...) {
    if (!diag || cap == 0) return;
    va_list ap; va_start(ap, fmt);
    vsnprintf(diag, cap, fmt, ap);
    va_end(ap);
}

/* skin-joint index of a node, or -1 if the node is not one of the skin's joints. */
static int node_to_jointset(const cgltf_skin *skin, const cgltf_node *node) {
    for (cgltf_size i = 0; i < skin->joints_count; ++i) if (skin->joints[i] == node) return (int)i;
    return -1;
}

/* DFS the joint forest (children in ascending skin-order) from the root, filling order[] with
 * skin-order indices parent-before-child. Returns the number of joints visited (== jc on success;
 * < jc signals a malformed hierarchy: a cycle or an unreachable joint). */
static uint32_t topo_order(const int32_t *parent_js, uint32_t jc, uint32_t root_js, uint32_t *order) {
    /* explicit stack to avoid recursion depth limits; emit children in ascending skin-order by
     * pushing them in descending order so the smallest is popped first */
    uint32_t *stack = (uint32_t *)malloc((size_t)jc * sizeof(uint32_t));
    if (!stack) return 0;
    uint32_t sp = 0, cnt = 0;
    stack[sp++] = root_js;
    while (sp) {
        uint32_t cur = stack[--sp];
        order[cnt++] = cur;
        /* push children (parent_js[child]==cur) in descending index order */
        for (uint32_t ch = jc; ch-- > 0; )
            if (parent_js[ch] == (int32_t)cur) {
                if (sp >= jc) { free(stack); return 0; } /* malformed (cycle) */
                stack[sp++] = ch;
            }
    }
    free(stack);
    return cnt;
}

mrw_result mrw_g2m_joint_order_build(const cgltf_skin *skin, mrw_g2m_joint_order *out,
                                     char *diag, size_t diag_cap) {
    if (!skin || !out) { set_diag(diag, diag_cap, "invalid arguments"); return MRW_E_FORMAT; }
    memset(out, 0, sizeof *out);

    if (skin->joints_count < 1 || skin->joints_count > 0xFFFEu) {
        set_diag(diag, diag_cap, "skin has %zu joints (must be 1..65534)", skin->joints_count);
        return MRW_E_RANGE;
    }
    uint32_t jc = (uint32_t)skin->joints_count;

    int32_t  *parent_js = (int32_t  *)malloc((size_t)jc * sizeof(int32_t));
    uint32_t *order     = (uint32_t *)malloc((size_t)jc * sizeof(uint32_t));
    uint32_t *marrow_of = (uint32_t *)malloc((size_t)jc * sizeof(uint32_t));
    if (!parent_js || !order || !marrow_of) {
        free(parent_js); free(order); free(marrow_of);
        set_diag(diag, diag_cap, "out of memory");
        return MRW_E_OVERFLOW;
    }

    /* parent-joint resolution + single-root requirement: each joint's parent is the nearest
     * ancestor node that is itself one of the skin's joints (intervening non-joint nodes are folded
     * over by the caller's pose math, not here). */
    int32_t root_js = -1; uint32_t root_count = 0;
    for (uint32_t i = 0; i < jc; ++i) {
        const cgltf_node *anc = skin->joints[i]->parent;
        int pj = -1;
        while (anc) { int idx = node_to_jointset(skin, anc); if (idx >= 0) { pj = idx; break; } anc = anc->parent; }
        parent_js[i] = pj;
        if (pj < 0) { root_js = (int32_t)i; ++root_count; }
    }
    if (root_count != 1) {
        free(parent_js); free(order); free(marrow_of);
        set_diag(diag, diag_cap, "skin has %u skeleton roots (marrow requires exactly 1)", root_count);
        return MRW_E_FORMAT;
    }

    /* topological order (parent before child) + its inverse map */
    if (topo_order(parent_js, jc, (uint32_t)root_js, order) != jc) {
        free(parent_js); free(order); free(marrow_of);
        set_diag(diag, diag_cap, "joint hierarchy is not a single tree");
        return MRW_E_FORMAT;
    }
    for (uint32_t mj = 0; mj < jc; ++mj) marrow_of[order[mj]] = mj;

    out->joint_count = jc;
    out->parent_js   = parent_js;
    out->order       = order;
    out->marrow_of   = marrow_of;
    out->root_js     = root_js;
    return MRW_OK;
}

void mrw_g2m_joint_order_free(mrw_g2m_joint_order *jo) {
    if (!jo) return;
    free(jo->parent_js);
    free(jo->order);
    free(jo->marrow_of);
    memset(jo, 0, sizeof *jo);
}
