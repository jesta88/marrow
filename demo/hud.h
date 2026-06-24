/* In-window HUD overlay: a screen-space text + frame-time-graph layer driven by the profiler.
 *
 * Text comes from the vendored stb_easy_font (demo/third_party), which emits quads; those are
 * triangulated on the CPU (no quad topology in core Vulkan) into a per-frame host vertex buffer and
 * drawn with a shader-object pipeline. Because shader-object state is sticky, the HUD draw issues a
 * FULL dynamic-state vector (reusing vkc_set_default_state) and then overrides depth/cull/blend for
 * an alpha-blended overlay - including an explicit color-blend EQUATION, which the default state
 * never sets. Recorded inside the BeginRendering scope, after the 3D draws. Demo-only. */
#ifndef DEMO_HUD_H
#define DEMO_HUD_H

#include "vk_context.h"
#include "profiler.h"

#include <stddef.h>

typedef struct {
    VkShaderEXT      vs, fs;
    VkPipelineLayout layout;
    /* one host vertex buffer per frame-in-flight (rebuilt each frame; sized for the worst case) */
    VkBuffer         vbuf[VKC_FRAMES_IN_FLIGHT];
    VkDeviceMemory   vmem[VKC_FRAMES_IN_FLIGHT];
    void            *vmap[VKC_FRAMES_IN_FLIGHT];
    uint32_t         max_verts;
    void            *quad_scratch;   /* stb_easy_font quad output (per-line) */
    size_t           quad_scratch_bytes;

    int              visible;        /* F1 */
    int              detail;         /* F2: 0 = summary, 1 = per-scope breakdown */
} Hud;

int  hud_init(Hud *h, VkCtx *ctx);
void hud_destroy(Hud *h, VkCtx *ctx);
/* Build the overlay geometry from `prof` and record the draw. No-op if !visible. Call after the 3D
 * draws and before vkc_end_frame (inside the active BeginRendering scope). */
void hud_draw(Hud *h, VkCtx *ctx, VkCommandBuffer cmd, VkExtent2D extent, const Profiler *prof);

/* One-line FPS/ms summary for the window title (free with the overlay off). */
void hud_title_line(const Profiler *prof, char *out, size_t n);

#endif /* DEMO_HUD_H */
