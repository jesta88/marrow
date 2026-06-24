#version 450
/* Tier-B crowd fragment: simple directional lighting over the per-instance tint. */

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 0) out vec4 oColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(vec3(0.4, 0.85, 0.35));
    float diff = max(dot(N, L), 0.0);
    float amb = 0.30;
    oColor = vec4(vColor * (amb + (1.0 - amb) * diff), 1.0);
}
