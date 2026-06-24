#version 450
/* Skeleton bone-line fragment: flat per-instance tint. Lines carry no meaningful surface normal, so
 * unlike crowd.frag there is no lighting term. */

layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 oColor;

void main() { oColor = vec4(vColor, 1.0); }
