#version 450
/* Screen-space HUD overlay. Positions arrive in PIXELS (origin top-left, +y down - the
 * stb_easy_font / Vulkan-framebuffer convention); a push-constant viewport size maps them to NDC. */

layout(push_constant) uniform PC { vec2 viewport; } pc;

layout(location = 0) in vec2 aPos;     /* pixels */
layout(location = 1) in vec4 aColor;

layout(location = 0) out vec4 vColor;

void main() {
    vec2 ndc = aPos / pc.viewport * 2.0 - 1.0;   /* [0,size] -> [-1,1], y down (Vulkan default) */
    gl_Position = vec4(ndc, 0.0, 1.0);
    vColor = aColor;
}
