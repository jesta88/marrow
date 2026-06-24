#version 450
/* Tier-B baked GPU skeleton: a lightweight bone-line render LOD for the crowd. Instead of skinning
 * ~432 mesh vertices per instance, it transforms two endpoints per bone (~2x bones), collapsing the
 * per-vertex skinning cost that bounds the full-mesh crowd at scale - so the frame-time stays green
 * while the per-instance/per-bone ANIMATION cost is unchanged.
 *
 * The sampling math (modPos / quatToRows / nlerp / decodeBone / sampleClip / resolveBone) is the
 * verbatim Tier-B path shared with crowd.vert. Per endpoint the vertex carries the joint index and
 * the joint's BIND-MODEL-SPACE origin; the posed joint origin is M_joint . restOrigin, where M is
 * the resolved baked bone. This rides the EXACT same palette the mesh skins from: the baked bone is
 * the skinning transform model_pose . inverse_bind, and restOrigin is bind_model . 0, so inverse_bind
 * cancels bind_model and M . restOrigin = model_pose . 0 = the posed joint origin. */
#extension GL_EXT_buffer_reference2     : require
#extension GL_EXT_nonuniform_qualifier  : require
#extension GL_EXT_scalar_block_layout    : require

struct InstanceAnim {
    mat4  model;
    uvec4 clipA;     /* first_frame, frame_count, looping, paletteIndex     */
    uvec4 clipB;     /* first_frame, frame_count, looping, _pad             */
    vec4  times;     /* tA, tB, durA, durB                                  */
    vec4  blend;     /* w, _, _, _                                          */
};
layout(buffer_reference, scalar) readonly buffer InstanceRef { InstanceAnim instances[]; };

/* Prefix of crowd.vert's CrowdPush (no bench field); compatible with the same pipeline layout. */
layout(push_constant, scalar) uniform PC {
    mat4        viewProj;
    InstanceRef ref;
    uint        texelsPerBone;   /* = 2 */
} pc;

layout(set = 0, binding = 0) uniform sampler2D uPalettes[];

layout(location = 0) in uint aJoint;        /* which joint's palette row transforms this endpoint */
layout(location = 1) in vec3 aRestOrigin;   /* the joint's bind-model-space origin                */

layout(location = 0) out vec3 vColor;

int gPalette;   /* bindless index of this instance's palette texture */

struct Xform { vec4 q; vec3 t; float s; };

float modPos(float t, float d) { float m = t - d * trunc(t / d); if (m < 0.0) m += d; if (m >= d) m -= d; return m; }
float clampf(float t, float lo, float hi) { return t < lo ? lo : (t > hi ? hi : t); }

void quatToRows(vec4 q, out vec3 r0, out vec3 r1, out vec3 r2) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float xx = x*x, yy = y*y, zz = z*z, xy = x*y, xz = x*z, yz = y*z, wx = w*x, wy = w*y, wz = w*z;
    r0 = vec3(1.0 - 2.0*(yy+zz), 2.0*(xy-wz),       2.0*(xz+wy));
    r1 = vec3(2.0*(xy+wz),       1.0 - 2.0*(xx+zz), 2.0*(yz-wx));
    r2 = vec3(2.0*(xz-wy),       2.0*(yz+wx),       1.0 - 2.0*(xx+yy));
}
vec4 renorm(vec4 q) { float n2 = dot(q, q); return q * (n2 > 0.0 ? 1.0 / sqrt(n2) : 0.0); }
vec4 nlerp(vec4 a, vec4 b, float u) { if (dot(a, b) < 0.0) b = -b; return renorm(mix(a, b, u)); }

Xform decodeBone(int bone, int F) {
    int x0 = bone * int(pc.texelsPerBone);
    vec4 t0 = texelFetch(uPalettes[nonuniformEXT(gPalette)], ivec2(x0,     F), 0);
    vec4 t1 = texelFetch(uPalettes[nonuniformEXT(gPalette)], ivec2(x0 + 1, F), 0);
    Xform o; o.q = renorm(t0); o.t = t1.xyz; o.s = t1.w; return o;
}

Xform sampleClip(int bone, uvec4 clip, float t, float dur) {
    int firstFrame = int(clip.x), frameCount = int(clip.y);
    bool looping = clip.z != 0u;
    int i0 = 0; float u = 0.0;
    if (frameCount != 1 && dur != 0.0) {
        float tl   = looping ? modPos(t, dur) : clampf(t, 0.0, dur);
        float fpos = (tl / dur) * float(frameCount - 1);
        int   i    = int(floor(fpos));
        if (i > frameCount - 2) i = frameCount - 2;
        i0 = i; u = fpos - float(i);
    }
    Xform a = decodeBone(bone, firstFrame + i0);
    if (u != 0.0) {
        Xform b = decodeBone(bone, firstFrame + i0 + 1);
        a.q = nlerp(a.q, b.q, u);
        a.t = mix(a.t, b.t, u);
        a.s = mix(a.s, b.s, u);
    }
    return a;
}

Xform resolveBone(int bone, InstanceAnim inst) {
    float w = inst.blend.x;
    if (w >= 1.0) return sampleClip(bone, inst.clipB, inst.times.y, inst.times.w);
    Xform xa = sampleClip(bone, inst.clipA, inst.times.x, inst.times.z);
    if (w > 0.0) {
        Xform xb = sampleClip(bone, inst.clipB, inst.times.y, inst.times.w);
        xa.q = nlerp(xa.q, xb.q, w);
        xa.t = mix(xa.t, xb.t, w);
        xa.s = mix(xa.s, xb.s, w);
    }
    return xa;
}

/* cheap per-instance tint for crowd variety - matches crowd.vert so a mesh and its skeleton agree */
vec3 tint(uint id) {
    return 0.45 + 0.45 * cos(vec3(float(id) * 0.91, float(id) * 1.73 + 2.0, float(id) * 2.21 + 4.0));
}

void main() {
    InstanceAnim inst = pc.ref.instances[gl_InstanceIndex];
    gPalette = int(inst.clipA.w);

    Xform xf = resolveBone(int(aJoint), inst);
    vec3 r0, r1, r2; quatToRows(xf.q, r0, r1, r2);
    vec3 rp = vec3(dot(r0, aRestOrigin), dot(r1, aRestOrigin), dot(r2, aRestOrigin));
    vec3 posed = xf.s * rp + xf.t;          /* M_joint . restOrigin = posed joint origin (model space) */

    vec4 worldPos = inst.model * vec4(posed, 1.0);
    gl_Position = pc.viewProj * worldPos;
    vColor = tint(uint(gl_InstanceIndex));
}
