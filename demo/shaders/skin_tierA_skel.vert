#version 450
/* Tier-A bone-line skeleton render LOD - the CPU-palette counterpart to crowd_skel.vert.
 *
 * Where skin_tierA.vert skins ~432 mesh vertices, this transforms two endpoints per bone (~2x bones):
 * the cheap render LOD that drops per-vertex cost off the vertex wall while the per-instance/per-bone
 * pose cost is unchanged. Per endpoint the vertex carries a joint index and that joint's
 * bind-model-space origin; the posed origin is M_joint . restOrigin, where M is the joint's canonical
 * Tier-A skinning palette row model_pose . inverse_bind (what mrw_model_to_palette writes). inverse_bind
 * cancels bind_model, so M . restOrigin = model_pose . 0 = the posed joint origin. The full-precision
 * f32 palette is fetched by index from the same BDA SSBO skin_tierA.vert reads - no bindless texture. */
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

layout(location = 0) in uint aJoint;        /* which joint's palette row poses this endpoint */
layout(location = 1) in vec3 aRestOrigin;   /* the joint's bind-model-space origin           */

layout(location = 0) out vec3 vColor;

void main() {
    HeroInstance inst = pc.heroes.h[gl_InstanceIndex];
    uint base = (uint(gl_InstanceIndex) * pc.jointCount + aJoint) * 3u;
    vec4 r0 = pc.palettes.rows[base + 0u];
    vec4 r1 = pc.palettes.rows[base + 1u];
    vec4 r2 = pc.palettes.rows[base + 2u];

    vec4 ro = vec4(aRestOrigin, 1.0);
    vec3 posed = vec3(dot(r0, ro), dot(r1, ro), dot(r2, ro));   /* M_joint . restOrigin = posed joint origin */

    vec4 world = inst.model * vec4(posed, 1.0);
    gl_Position = pc.viewProj * world;
    vColor = inst.tint.rgb;
}
