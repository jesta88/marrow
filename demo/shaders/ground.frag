#version 450
/* Antialiased grid floor with brighter major lines every 10 m, fading with distance. */

layout(location = 0) in vec3 vWorld;
layout(location = 0) out vec4 oColor;

float gridLine(vec2 p) {
    vec2 g = abs(fract(p) - 0.5) / fwidth(p);
    return 1.0 - min(min(g.x, g.y), 1.0);
}

void main() {
    vec3 base = vec3(0.09, 0.10, 0.12);
    float minor = gridLine(vWorld.xz);
    float major = gridLine(vWorld.xz * 0.1);
    vec3 col = mix(base, vec3(0.22, 0.25, 0.30), minor);
    col = mix(col, vec3(0.45, 0.55, 0.65), major);
    float d = length(vWorld.xz);
    col = mix(col, base, clamp((d - 20.0) / 60.0, 0.0, 1.0));
    oColor = vec4(col, 1.0);
}
