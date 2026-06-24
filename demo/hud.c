#include "hud.h"

#include "third_party/stb_easy_font.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include "hud_vert.spv.h"
#include "hud_frag.spv.h"

/* One screen-space HUD vertex (pixels + RGBA float). Mirrors hud.vert's vertex input. */
typedef struct { float pos[2]; float col[4]; } HudVertex;

/* stb_easy_font quad vertex: x,y,z float + 4 color bytes (16 bytes, interleaved). */
typedef struct { float x, y, z; unsigned char c[4]; } StbVert;
_Static_assert(sizeof(StbVert) == 16, "stb_easy_font vertex is 16 bytes");

#define HUD_MAX_VERTS   49152u   /* triangulated overlay geometry (worst case, generous) */
#define HUD_SCRATCH     65536u   /* stb_easy_font quad output per line */
#define HUD_SCALE       2.0f
#define HUD_LINE_STEP   27.0f    /* 12px stb cell * scale + gap */
#define HUD_PAD         8.0f
#define HUD_GRAPH_H     46.0f
#define HUD_GRAPH_TGT   33.34f   /* ms at full bar height (30 fps line) */

static const char *k_cpu_name[PROF_CPU_COUNT] = {
    "wait", "acquire", "crowd", "heroes", "palette_gen", "upload", "record", "submit", "lod"
};
static const char *k_gpu_name[PROF_GPU_COUNT] = { "frame", "ground", "crowd", "heroes", "skel", "hud" };

/* ------------------------------------------------------------------ geometry builder */

typedef struct { HudVertex *v; uint32_t n, cap; } Builder;

static void push_v(Builder *b, float x, float y, const float c[4]) {
    if (b->n >= b->cap) return;
    HudVertex *o = &b->v[b->n++];
    o->pos[0] = x; o->pos[1] = y;
    o->col[0] = c[0]; o->col[1] = c[1]; o->col[2] = c[2]; o->col[3] = c[3];
}

static void emit_tri(Builder *b, float x0, float y0, float x1, float y1,
                     float x2, float y2, const float c[4]) {
    push_v(b, x0, y0, c); push_v(b, x1, y1, c); push_v(b, x2, y2, c);
}

static void emit_rect(Builder *b, float x0, float y0, float x1, float y1, const float c[4]) {
    emit_tri(b, x0, y0, x1, y0, x1, y1, c);
    emit_tri(b, x0, y0, x1, y1, x0, y1, c);
}

/* Render one text line via stb_easy_font, scaled, at (x,y) top-left in pixels. */
static void emit_text(Hud *h, Builder *b, float x, float y, const float c[4], const char *s) {
    int nq = stb_easy_font_print(0.0f, 0.0f, (char *)s, NULL, h->quad_scratch, (int)h->quad_scratch_bytes);
    const StbVert *q = (const StbVert *)h->quad_scratch;
    for (int i = 0; i < nq; ++i) {
        const StbVert *v = &q[i * 4];
        float px[4], py[4];
        for (int k = 0; k < 4; ++k) { px[k] = x + v[k].x * HUD_SCALE; py[k] = y + v[k].y * HUD_SCALE; }
        emit_tri(b, px[0], py[0], px[1], py[1], px[2], py[2], c);
        emit_tri(b, px[0], py[0], px[2], py[2], px[3], py[3], c);
    }
}

/* ------------------------------------------------------------------ init / destroy */

int hud_init(Hud *h, VkCtx *ctx) {
    memset(h, 0, sizeof *h);
    h->max_verts = HUD_MAX_VERTS;
    h->quad_scratch_bytes = HUD_SCRATCH;
    h->quad_scratch = malloc(h->quad_scratch_bytes);
    if (!h->quad_scratch) { fprintf(stderr, "[hud] scratch alloc failed\n"); return 0; }

    VkPushConstantRange pr = { VK_SHADER_STAGE_VERTEX_BIT, 0, 2 * sizeof(float) };
    VkPipelineLayoutCreateInfo li = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(ctx->device, &li, NULL, &h->layout) != VK_SUCCESS) {
        fprintf(stderr, "[hud] pipeline layout failed\n"); goto fail;
    }
    if (!vkc_create_graphics_shaders(ctx, hud_vert_spv, sizeof hud_vert_spv,
            hud_frag_spv, sizeof hud_frag_spv, NULL, 0, &pr, &h->vs, &h->fs)) goto fail;

    VkDeviceSize bytes = (VkDeviceSize)h->max_verts * sizeof(HudVertex);
    for (uint32_t f = 0; f < VKC_FRAMES_IN_FLIGHT; ++f)
        if (!vkc_create_host_buffer(ctx, bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                    &h->vbuf[f], &h->vmem[f], &h->vmap[f])) goto fail;
    h->visible = 1;
    return 1;
fail:   /* destroy unwinds whatever was created; the init-time memset left the rest NULL */
    hud_destroy(h, ctx);
    return 0;
}

void hud_destroy(Hud *h, VkCtx *ctx) {
    vkc_destroy_shader(ctx, h->vs); vkc_destroy_shader(ctx, h->fs);
    if (h->layout) vkDestroyPipelineLayout(ctx->device, h->layout, NULL);
    for (uint32_t f = 0; f < VKC_FRAMES_IN_FLIGHT; ++f)
        vkc_destroy_buffer(ctx, h->vbuf[f], h->vmem[f]);
    free(h->quad_scratch);
    memset(h, 0, sizeof *h);
}

/* ------------------------------------------------------------------ per-frame build + draw */

static double frame_ms_now(const Profiler *p) { return p->fps > 0.0 ? 1000.0 / p->fps : 0.0; }

static void build_graph(const Profiler *p, Builder *b, float gx, float gy, float gw, float gh) {
    const float bg[4]   = { 0.0f, 0.0f, 0.0f, 0.50f };
    const float line[4] = { 0.45f, 0.50f, 0.60f, 0.6f };
    emit_rect(b, gx, gy, gx + gw, gy + gh, bg);
    /* 16.7 ms reference line (60 fps) */
    float ref_y = gy + gh - (16.67f / HUD_GRAPH_TGT) * gh;
    emit_rect(b, gx, ref_y, gx + gw, ref_y + 1.0f, line);

    if (!p->frame_count) return;
    float bw = gw / (float)PROF_FRAMES;
    uint32_t start = (p->frame_head + PROF_FRAMES - p->frame_count) % PROF_FRAMES;
    for (uint32_t i = 0; i < p->frame_count; ++i) {
        float ms = p->frame_ms[(start + i) % PROF_FRAMES];
        float frac = ms / HUD_GRAPH_TGT; if (frac > 1.0f) frac = 1.0f; if (frac < 0.0f) frac = 0.0f;
        float bh = frac * gh;
        const float green[4]  = { 0.40f, 0.90f, 0.45f, 0.95f };
        const float yellow[4] = { 0.95f, 0.85f, 0.30f, 0.95f };
        const float red[4]    = { 0.95f, 0.40f, 0.35f, 0.95f };
        const float *col = ms <= 16.8f ? green : (ms <= 33.4f ? yellow : red);
        float bx = gx + (float)i * bw;
        emit_rect(b, bx, gy + gh - bh, bx + (bw > 1.0f ? bw - 0.5f : bw), gy + gh, col);
    }
}

void hud_draw(Hud *h, VkCtx *ctx, VkCommandBuffer cmd, VkExtent2D extent, const Profiler *prof) {
    if (!h->visible) return;

    /* ---- assemble text lines ---- (sized for the full F2-detail list: header + per-zone CPU/GPU rows) */
    char L[32][128]; float C[32][4]; int n = 0;
    const float white[4] = { 0.92f, 0.94f, 1.00f, 1.0f };
    const float gold[4]  = { 1.00f, 0.82f, 0.20f, 1.0f };
    const float dim[4]   = { 0.70f, 0.76f, 0.88f, 1.0f };
    #define LINE(c, ...) do { if (n < 32) { snprintf(L[n], sizeof L[0], __VA_ARGS__); \
        C[n][0]=(c)[0]; C[n][1]=(c)[1]; C[n][2]=(c)[2]; C[n][3]=(c)[3]; ++n; } } while (0)

    LINE(gold, "marrow demo   %s", prof->model ? prof->model : "?");
    LINE(dim,  "  %s / %s", prof->tier, prof->backend);
    LINE(white, "FPS %5.1f    frame %.2f ms", prof->fps, frame_ms_now(prof));
    LINE(white, "CPU work %.2f ms  (p95 %.2f, waits excl.)",
         prof->cpu_total.ema, prof_pct(&prof->cpu_total, 0.95));
    if (prof->gpu_supported)
        LINE(white, "GPU      %.2f ms  (p95 %.2f)",
             prof->gpu[PROF_GPU_FRAME].ema, prof_pct(&prof->gpu[PROF_GPU_FRAME], 0.95));
    else
        LINE(white, "GPU      n/a (no timestamps)");
    LINE(dim, "  palette gen %.3f   upload %.3f ms",
         prof->cpu[PROF_PALETTE_GEN].ema, prof->cpu[PROF_PALETTE_UPLOAD].ema);
    /* Scale-invariant kernel throughput for the CPU tier: PALETTE_GEN normalized by the bone-
     * instances it covered. Directly comparable across backends and entity counts (the headline
     * mrw_batch number the validate microbench also reports). */
    if (prof->gen_bones)
        LINE(gold, "  kernel %.2f ns/(inst.bone)  [%s]",
             prof->cpu[PROF_PALETTE_GEN].ema * 1e6 / (double)prof->gen_bones, prof->backend);
    LINE(dim, "inst %llu  draws %llu  tris %.2fM  lines %.2fM  bones %.2fM",
         (unsigned long long)prof->instances, (unsigned long long)prof->draws,
         (double)prof->triangles * 1e-6, (double)prof->lines * 1e-6, (double)prof->bones * 1e-6);
    if (prof->field_r_a > 0.0f) {
        LINE(gold, "LOD  near(A) %u  far(B) %u  [mesh %u / skel %u]  lod %.3f ms",
             prof->field_near, prof->field_far, prof->field_far_mesh, prof->field_far_skel,
             prof->cpu[PROF_LOD].ema);
        LINE(dim, "  R_A %.0f  R_mesh %.0f%s", (double)prof->field_r_a, (double)prof->field_r_mesh,
             prof->field_skel_all ? "   K: all-skeleton" : "");
        if (prof->field_clamped)
            LINE(gold, "  near-cap clamp: %u entities exceed near_cap -> Tier B", prof->field_clamped);
    }

    if (h->detail) {
        LINE(gold, "-- CPU scopes (ms, EMA) --");
        for (int z = 0; z < PROF_CPU_COUNT; ++z)
            LINE(dim, "  %-12s %.3f", k_cpu_name[z], prof->cpu[z].ema);
        if (prof->gpu_supported) {
            LINE(gold, "-- GPU zones (ms, EMA) --");
            for (int z = 0; z < PROF_GPU_COUNT; ++z)
                LINE(dim, "  %-12s %.3f", k_gpu_name[z], prof->gpu[z].ema);
        }
    }
    if (prof->field_r_a > 0.0f)
        LINE(dim, "F1 hud  F2 detail  M model  -/= count  [ ] R_A  ; ' R_mesh  K skel  R reset");
    else
        LINE(dim, "F1 hud  F2 detail  M model  T tier  B/123 backend  J jobs  H f16  -/= count  R reset  P promote");
    #undef LINE

    /* ---- panel + text + graph geometry ---- */
    Builder b = { (HudVertex *)h->vmap[ctx->cur_frame], 0, h->max_verts };

    float max_w = 0.0f;
    for (int i = 0; i < n; ++i) {
        float w = (float)stb_easy_font_width(L[i]) * HUD_SCALE;
        if (w > max_w) max_w = w;
    }
    float panel_w = max_w + 2.0f * HUD_PAD;
    float text_h  = (float)n * HUD_LINE_STEP;
    float panel_h = HUD_PAD + text_h + HUD_PAD + HUD_GRAPH_H + HUD_PAD;
    float px = 6.0f, py = 6.0f;

    const float panel[4] = { 0.04f, 0.05f, 0.07f, 0.72f };
    emit_rect(&b, px, py, px + panel_w, py + panel_h, panel);

    float ty = py + HUD_PAD;
    for (int i = 0; i < n; ++i) { emit_text(h, &b, px + HUD_PAD, ty, C[i], L[i]); ty += HUD_LINE_STEP; }

    build_graph(prof, &b, px + HUD_PAD, ty, panel_w - 2.0f * HUD_PAD, HUD_GRAPH_H);

    if (b.n == 0) return;

    /* ---- record the overlay draw ---- */
    vkc_gpu_zone_begin(ctx, cmd, VKC_GPU_ZONE_HUD);
    VkVertexInputBindingDescription2EXT bind = { VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT };
    bind.binding = 0; bind.stride = sizeof(HudVertex); bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; bind.divisor = 1;
    VkVertexInputAttributeDescription2EXT attrs[2];
    for (int i = 0; i < 2; ++i) {
        attrs[i] = (VkVertexInputAttributeDescription2EXT){ VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT };
        attrs[i].location = (uint32_t)i; attrs[i].binding = 0;
    }
    attrs[0].format = VK_FORMAT_R32G32_SFLOAT;       attrs[0].offset = offsetof(HudVertex, pos);
    attrs[1].format = VK_FORMAT_R32G32B32A32_SFLOAT; attrs[1].offset = offsetof(HudVertex, col);

    vkc_bind_shaders(ctx, cmd, h->vs, h->fs);
    /* Shader-object state is sticky and the last 3D draw left depth ON / blend OFF, so issue the
     * FULL default state vector, then override for a 2D alpha-blended overlay. */
    vkc_set_default_state(ctx, cmd, extent, &bind, 1, attrs, 2);
    vkCmdSetDepthTestEnable(cmd, VK_FALSE);
    vkCmdSetDepthWriteEnable(cmd, VK_FALSE);
    vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
    VkBool32 blend_on = VK_TRUE;
    vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blend_on);
    /* vkc_set_default_state never sets a blend EQUATION - supply it explicitly (premultiply-free
     * src-alpha over). */
    VkColorBlendEquationEXT eq = {
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp        = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .alphaBlendOp        = VK_BLEND_OP_ADD,
    };
    vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &eq);

    float vp[2] = { (float)extent.width, (float)extent.height };
    vkCmdPushConstants(cmd, h->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof vp, vp);
    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &h->vbuf[ctx->cur_frame], &zero);
    vkCmdDraw(cmd, b.n, 1, 0, 0);
    vkc_gpu_zone_end(ctx, cmd, VKC_GPU_ZONE_HUD);
}

void hud_title_line(const Profiler *prof, char *out, size_t n) {
    const char *model = prof->model ? prof->model : "?";
    if (prof->gpu_supported)
        snprintf(out, n, "marrow demo - %s | %.0f fps | CPU %.2f ms | GPU %.2f ms | %s/%s",
                 model, prof->fps, prof->cpu_total.ema, prof->gpu[PROF_GPU_FRAME].ema, prof->tier, prof->backend);
    else
        snprintf(out, n, "marrow demo - %s | %.0f fps | CPU %.2f ms | GPU n/a | %s/%s",
                 model, prof->fps, prof->cpu_total.ema, prof->tier, prof->backend);
}
