#version 450
/* Tier-B baked GPU crowd skinning - Vulkan-GLSL adaptation of examples/skinning/marrow_skin.glsl.
 *
 * The sampling math (modPos / quatToRows / nlerp / decodeBone / sampleClip / resolveBone) is a
 * verbatim transcription of the gated CPU path (src/marrow_pose.c mrw_baked_sample_bone) and mirrors
 * examples/skinning/marrow_skin.glsl. Only the I/O plumbing differs from the OpenGL reference: per-instance
 * state via GL_EXT_buffer_reference2, the palette via a bindless sampler2D[] array, gl_InstanceIndex.
 *
 * Tier B is a FROZEN cache of Tier A: decode Q+T+uniform-scale, renormalize, §3.4 temporal nlerp/lerp
 * + the sanctioned <=2-clip cross-fade, §3.2 linear-blend skin. The normal uses s²·R - valid ONLY
 * because baked data is uniform-scale (Tier A uses the general cofactor; see skin_tierA.vert). */
#extension GL_EXT_buffer_reference2     : require
#extension GL_EXT_nonuniform_qualifier  : require
#extension GL_EXT_scalar_block_layout    : require

#define MARROW_INFLUENCES 4

struct InstanceAnim {
    mat4  model;     /* bind->world; CPU root motion folded in              */
    uvec4 clipA;     /* first_frame, frame_count, looping, paletteIndex     */
    uvec4 clipB;     /* first_frame, frame_count, looping, _pad             */
    vec4  times;     /* tA, tB, durA, durB                                  */
    vec4  blend;     /* w, _, _, _                                          */
};
layout(buffer_reference, scalar) readonly buffer InstanceRef { InstanceAnim instances[]; };

layout(push_constant, scalar) uniform PC {
    mat4        viewProj;
    InstanceRef ref;
    uint        texelsPerBone;   /* §7.5 = 2 */
#ifdef MARROW_BENCH
    float       benchEps;        /* bench-only: folds skin output into gl_Position so rasterizer
                                    discard can't DCE the skinning (value irrelevant - raster is off) */
#endif
} pc;

/* Bindless palette textures (one 2D RGBA16F per skeleton block); integer fetch only, no filtering. */
layout(set = 0, binding = 0) uniform sampler2D uPalettes[];

layout(location = 0) in vec3  aPos;
layout(location = 1) in vec3  aNormal;
layout(location = 2) in vec4  aTangent;
layout(location = 3) in uvec4 aBones;
layout(location = 4) in vec4  aWeights;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;

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

/* cheap per-instance tint for crowd variety */
vec3 tint(uint id) {
    return 0.45 + 0.45 * cos(vec3(float(id) * 0.91, float(id) * 1.73 + 2.0, float(id) * 2.21 + 4.0));
}

void main() {
    InstanceAnim inst = pc.ref.instances[gl_InstanceIndex];
    gPalette = int(inst.clipA.w);

    vec3 posAccum = vec3(0.0), nrmAccum = vec3(0.0);
    for (int i = 0; i < MARROW_INFLUENCES; ++i) {
        float wt = aWeights[i];
        if (wt == 0.0) continue;
        Xform xf = resolveBone(int(aBones[i]), inst);
        vec3 r0, r1, r2; quatToRows(xf.q, r0, r1, r2);
        vec3 rp = vec3(dot(r0, aPos), dot(r1, aPos), dot(r2, aPos));
        posAccum += wt * (xf.s * rp + xf.t);
        vec3 rn = vec3(dot(r0, aNormal), dot(r1, aNormal), dot(r2, aNormal));
        nrmAccum += wt * (xf.s * xf.s) * rn;          /* Tier-B s²·R (uniform-scale baked) */
    }

    vec4 worldPos = inst.model * vec4(posAccum, 1.0);
    gl_Position = pc.viewProj * worldPos;
#ifdef MARROW_BENCH
    /* Vertex-skinning microbenchmark (rasterizer discard on): pos already feeds gl_Position; fold
     * nrmAccum in via a runtime factor so the compiler keeps the WHOLE skinning. No fragment outputs. */
    gl_Position += pc.benchEps * vec4(nrmAccum, 0.0);
#else
    mat3 m3 = mat3(inst.model);
    vNormal = normalize(m3 * nrmAccum);
    vColor = tint(uint(gl_InstanceIndex));
#endif
}
