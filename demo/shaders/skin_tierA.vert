#version 450
/* Tier-A GPU skinning - the LOD-promotion counterpart to crowd.vert.
 *
 * Tier A computes the skinning palette on the CPU (marrow: sample -> engine cross-fade ->
 * local->model -> model_to_palette) and uploads the canonical 3x4 affines; this shader does only
 * the linear-blend skin. Because mrw_model_to_palette yields a GENERAL affine (the hierarchy may
 * carry non-uniform scale), the normal uses the 3x3 COFACTOR cof(A) = det(A)·A^-T - NOT Tier-B's
 * s²·R (valid only for uniform-scale baked data). cof(A) rows are the cross products of A's rows,
 * mirroring src/marrow_math.c mrw_affine_cofactor. */
#extension GL_EXT_buffer_reference2  : require
#extension GL_EXT_scalar_block_layout : require

struct HeroInstance { mat4 model; vec4 tint; };
layout(buffer_reference, scalar) readonly buffer HeroBuf    { HeroInstance h[]; };
layout(buffer_reference, scalar) readonly buffer PaletteBuf { vec4 rows[]; };   /* 3 rows (3x4) per (hero,joint) */

layout(push_constant, scalar) uniform PC {
    mat4       viewProj;
    HeroBuf    heroes;
    PaletteBuf palettes;
    uint       jointCount;
} pc;

layout(location = 0) in vec3  aPos;
layout(location = 1) in vec3  aNormal;
layout(location = 2) in vec4  aTangent;
layout(location = 3) in uvec4 aBones;
layout(location = 4) in vec4  aWeights;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;

void main() {
    HeroInstance inst = pc.heroes.h[gl_InstanceIndex];
    uint paletteBase = uint(gl_InstanceIndex) * pc.jointCount;

    vec3 pos = vec3(0.0), nrm = vec3(0.0);
    vec4 ph = vec4(aPos, 1.0);
    for (int i = 0; i < 4; ++i) {
        float wt = aWeights[i];
        if (wt == 0.0) continue;
        uint base = (paletteBase + aBones[i]) * 3u;
        vec4 r0 = pc.palettes.rows[base + 0u];
        vec4 r1 = pc.palettes.rows[base + 1u];
        vec4 r2 = pc.palettes.rows[base + 2u];

        pos += wt * vec3(dot(r0, ph), dot(r1, ph), dot(r2, ph));      /* M·[p,1] */

        /* cof(A) rows = r1×r2, r2×r0, r0×r1 (A = the 3x3 basis part) */
        vec3 a0 = r0.xyz, a1 = r1.xyz, a2 = r2.xyz;
        vec3 cof = vec3(dot(cross(a1, a2), aNormal),
                        dot(cross(a2, a0), aNormal),
                        dot(cross(a0, a1), aNormal));
        nrm += wt * cof;
    }

    vec4 world = inst.model * vec4(pos, 1.0);
    gl_Position = pc.viewProj * world;
    vNormal = normalize(mat3(inst.model) * nrm);
    vColor  = inst.tint.rgb;
}
