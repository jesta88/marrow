#version 450
/* Tier-A GPU skinning - f16 palette variant of skin_tierA.vert.
 *
 * Identical math to skin_tierA.vert; the only change is the palette FETCH. The CPU batch
 * (mrw_batch_clip_to_palette_f16) emits the SAME canonical 3x4 affine, components narrowed to
 * binary16 - 24 B/joint instead of 48. There is no quaternion here (this is the already-composed
 * matrix, not a Tier-B Q+T cache), so there is no renormalize: it is a DIRECT fetch + LBS.
 *
 * The device exposes scalarBlockLayout + BDA but not 16-bit storage, so the palette is read as
 * packed uints and widened with unpackHalf2x16 (a core built-in, no extra feature). Each joint is
 * 3 uvec2 rows = 12 binary16; unpackHalf2x16 maps the low/high 16 bits to .x/.y, matching the
 * little-endian [A00,A01,A02,tx, A10,...] component order the kernel writes. Reading the SSBO by
 * index is an integer fetch with no filtering - exactly what the palette decode requires. */
#extension GL_EXT_buffer_reference2  : require
#extension GL_EXT_scalar_block_layout : require

struct HeroInstance { mat4 model; vec4 tint; };
layout(buffer_reference, scalar) readonly buffer HeroBuf    { HeroInstance h[]; };
layout(buffer_reference, scalar) readonly buffer PaletteBuf { uvec2 rows[]; };   /* 3 packed rows (3x4) per (hero,joint) */

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

/* widen one packed row (2 uints = 4 binary16) to a full-precision vec4 */
vec4 unpackRow(uvec2 p) { return vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y)); }

void main() {
    HeroInstance inst = pc.heroes.h[gl_InstanceIndex];
    uint paletteBase = uint(gl_InstanceIndex) * pc.jointCount;

    vec3 pos = vec3(0.0), nrm = vec3(0.0);
    vec4 ph = vec4(aPos, 1.0);
    for (int i = 0; i < 4; ++i) {
        float wt = aWeights[i];
        if (wt == 0.0) continue;
        uint base = (paletteBase + aBones[i]) * 3u;
        vec4 r0 = unpackRow(pc.palettes.rows[base + 0u]);
        vec4 r1 = unpackRow(pc.palettes.rows[base + 1u]);
        vec4 r2 = unpackRow(pc.palettes.rows[base + 2u]);

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
