#version 450
/* Phase-0 ground grid: a single large quad at y=0, grid drawn in the fragment shader. */

layout(push_constant) uniform PC { mat4 viewProj; } pc;

layout(location = 0) in vec3 aPos;
layout(location = 0) out vec3 vWorld;

void main() {
    vWorld = aPos;
    gl_Position = pc.viewProj * vec4(aPos, 1.0);
}
