#version 450
/* HUD overlay: emit the interpolated vertex color straight through (alpha-blended by the caller). */

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 oColor;

void main() { oColor = vColor; }
