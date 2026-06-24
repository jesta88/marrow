#version 450
/* marrow - reference Tier-B GPU skinning vertex shader (GLSL 450, canonical reference).
 *
 * Consumes the baked per-bone palette stream (marrow .mrw BAKED section, encoding 1:
 * Q+T+uniform-scale, 2 RGBA16F texels/bone, frame-major). It is a *faithful transcription* of the
 * already-gated CPU path - each step cross-references the C source it mirrors:
 *   decode/renorm + frame pick + nlerp/lerp + compose  ->  src/marrow_pose.c  mrw_baked_sample_bone
 *   the scalar helpers                                 ->  src/marrow_internal.h
 *   quat->R and TRS->3x4 (A = s·R)                     ->  src/marrow_math.c
 *   component-space oracle (decode/sample/cross-nlerp)  ->  tools/bake/mrw_bake.c
 * tests/test_parity.c gates that CPU math against Tier A. The texture-upload contract, the I/O
 * contract, and the verification recipe are documented inline below.
 *
 * GUARDRAIL: Tier B is a FROZEN cache of Tier A. This shader does NOT evaluate the
 * skeleton hierarchy, IK, masks, or a blend graph; it does NOT multiply by inverse-bind (already
 * pre-applied into the palette matrix M); and it does NOT apply root motion (excluded from
 * the palette - the engine evaluates the CLIP root_track on the CPU and folds it into the
 * per-instance model matrix below). Anything richer PROMOTES the entity to Tier A on the CPU.
 *
 * This file is reference material, NOT part of the zero-dependency runtime and NOT built by CMake. */

/* ------------------------------------------------------------------ feature knobs (LOD)
 * The DEFAULT path (all knobs at their reference values) is the exact runtime reference math and matches
 * mrw_baked_sample_bone. The reduced settings are *approximate* distant-crowd LODs (distant LODs
 * dial down influences, drop temporal interpolation, and skip cross-fade) and are
 * explicitly NOT equal to the runtime - do not treat them as the reference math. */
#define MARROW_INFLUENCES        4   /* bones/vertex (reference 4). Fewer = approximate.            */
#define MARROW_TEMPORAL_INTERP   1   /* 1 = nlerp/lerp between frames (reference).                  */
                                     /* 0 = hold previous (floor) frame - APPROXIMATE, no interp.  */
#define MARROW_CROSSFADE         1   /* 1 = sanctioned ≤2-clip cross-fade. 0 = single clip.         */

/* GPU fetch budget (worst case): influences × 2 texels × 2 frames × 2 clips.
 *   reference (4,interp,fade) = 4·2·2·2 = 32 fetches/vertex (the CEILING, not the norm).
 *   no fade                   = 4·2·2·1 = 16;  no temporal interp = 4·2·1·2 = 16;  both = 8. */

/* ------------------------------------------------------------------ bindings
 * Per-INSTANCE animation state (crowd-scale): one entry per drawn entity, indexed by the instance
 * id. marrow's reason for existing is tens of thousands of entities, so animation state is per
 * INSTANCE, never per-draw uniforms. The SSBO transport shown here is ENGINE POLICY, not part of
 * the contract - an engine may instead feed this state as per-instance (instanced) vertex
 * attributes. Either way it feeds the same pure math functions below. */
struct InstanceAnim {
    mat4 model;     /* bind/model space -> WORLD (engine folds CPU-evaluated root motion in here) */
    uvec4 clipA;    /* (first_frame, frame_count, looping(0/1), _pad)   - baked clip-table entry A */
    uvec4 clipB;    /* (first_frame, frame_count, looping(0/1), _pad)   - entry B (cross-fade)     */
    vec4  times;    /* (tA, tB, source_duration_A, source_duration_B)   - clip-local seconds + dur */
    vec4  blend;    /* (w, _, _, _)   - cross-fade weight w ∈ [0,1]; w==0 ⇒ clip A only            */
};
layout(std430, binding = 1) readonly buffer InstanceBlock { InstanceAnim instances[]; };

/* The baked palette, packed by the engine into an RGBA16F 2D texture (atlas packing is renderer
 * policy). Sampled ONLY via texelFetch (integer, nearest, NO filtering): the decode requires
 * hemisphere-corrected nlerp in COMPONENT space, and hardware bilinear would both lerp quaternions
 * wrongly and bleed across bone/frame texel boundaries. */
layout(binding = 0) uniform sampler2D uPalette;

/* Per-fetch addressing inputs. This reference implements the RECOMMENDED 2D-block packing:
 * the palette is a 2D texture (or a 2D atlas) where each skeleton block is `frame_stride_texels`
 * texels wide × `total_frames` rows, so a frame is one texture row and a bone is `texels_per_bone`
 * texels across. The shader therefore addresses by (origin + bone·texels_per_bone + c, origin + F)
 * and needs only `texels_per_bone` + the block origin. `frame_stride_texels` is an UPLOAD/PACKING
 * parameter (the block's row width), NOT a per-fetch shader input here. A LINEAR /
 * row-wrapping buffer (frames not aligned to texture rows) would instead use the linear
 * address with explicit stride; an ARRAY atlas would add a sampler2DArray layer.
 * uPaletteOrigin is a per-DRAW uniform because an instanced draw is one skeleton; a heterogeneous
 * atlas would move it per-instance. */
layout(location = 0) uniform int   uTexelsPerBone;  /* = 2 (q texel + t/s texel)                   */
layout(location = 1) uniform ivec2 uPaletteOrigin;  /* atlas origin (x,y) of this skeleton block   */
layout(location = 2) uniform mat4  uViewProj;       /* world -> clip (occupies locations 2..5)     */

/* ------------------------------------------------------------------ vertex inputs (bind space) */
layout(location = 0) in vec3  aPos;       /* bind-pose position                                   */
layout(location = 1) in vec3  aNormal;    /* bind-pose normal                                     */
layout(location = 2) in vec4  aTangent;   /* bind-pose tangent; .w = bitangent handedness (±1)    */
layout(location = 3) in ivec4 aBones;     /* up to 4 palette bone indices                         */
layout(location = 4) in vec4  aWeights;   /* skin weights - ASSUMED pre-normalized (Σ=1, glTF);   */
                                          /* the reference does NOT renormalize (engine's job).   */

/* ------------------------------------------------------------------ outputs */
layout(location = 0) out vec3 vNormal;    /* world-space normal                                   */
layout(location = 1) out vec4 vTangent;   /* world-space tangent.xyz + preserved handedness .w    */

/* component-space transform of one bone at one phase (pre-compose) */
struct Xform { vec4 q; vec3 t; float s; };

/* ------------------------------------------------------------------ scalar helpers
 * Implemented EXPLICITLY (not via GLSL builtins) to match the C bit-for-bit. */

/* mrw_clampf (src/marrow_internal.h). */
float clampf(float t, float lo, float hi) { return t < lo ? lo : (t > hi ? hi : t); }

/* mrw_mod_pos: ((t mod d)+d) mod d kept strictly in [0,d). GLSL has no truncated fmod, so the
 * remainder is t − d·trunc(t/d); then BOTH guards - the +d can round to exactly d in float, so the
 * trailing subtract re-applies the outer modulo. Do NOT substitute the floored builtin mod(). */
float modPos(float t, float d) {
    float m = t - d * trunc(t / d);
    if (m < 0.0) m += d;
    if (m >= d) m -= d;
    return m;
}

/* Unit quaternion q=(x,y,z,w) -> 3×3 rotation, returned as ROW vectors (column-vector v'=R·v).
 * Rows (not a mat3 - GLSL mat3 constructors are COLUMN-major, a transpose trap). Mirrors
 * mrw_quat_to_mat3. */
void quatToRows(vec4 q, out vec3 r0, out vec3 r1, out vec3 r2) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;
    r0 = vec3(1.0 - 2.0*(yy+zz),  2.0*(xy-wz),        2.0*(xz+wy));
    r1 = vec3(2.0*(xy+wz),        1.0 - 2.0*(xx+zz),  2.0*(yz-wx));
    r2 = vec3(2.0*(xz-wy),        2.0*(yz+wx),        1.0 - 2.0*(xx+yy));
}

/* full-precision quaternion renormalize: inv = 1/sqrt(|q|²) (or 0), matching the C exactly - NOT
 * inversesqrt(), whose hardware approximation would drift from the oracle. */
vec4 renorm(vec4 q) {
    float n2 = dot(q, q);
    return q * (n2 > 0.0 ? 1.0 / sqrt(n2) : 0.0);
}

/* Hemisphere-corrected nlerp. Flip b on dot < 0.0 (the value compare, NOT the raw sign bit:
 * the oracle does not flip on dot == −0.0). Mirrors mrw_quat_nlerp. */
vec4 nlerp(vec4 a, vec4 b, float u) {
    if (dot(a, b) < 0.0) b = -b;
    return renorm(mix(a, b, u));   /* mix(a,b,u) = a + u·(b−a) */
}

/* Decode of one bone at one ABSOLUTE baked frame F: two integer texel fetches (no filtering),
 * renormalize q. Texel address: texel0=(qx,qy,qz,qw), texel1=(tx,ty,tz,s). Mirrors
 * baked_decode_bone / mrw_bake_decode. */
Xform decodeBone(int bone, int F) {
    int x0 = uPaletteOrigin.x + bone * uTexelsPerBone;   /* b·texels_per_bone (+c)                */
    int y  = uPaletteOrigin.y + F;                       /*       absolute frame row              */
    vec4 t0 = texelFetch(uPalette, ivec2(x0,     y), 0); /* c=0: q                                */
    vec4 t1 = texelFetch(uPalette, ivec2(x0 + 1, y), 0); /* c=1: (t, s)                           */
    Xform o;
    o.q = renorm(t0);
    o.t = t1.xyz;
    o.s = t1.w;
    return o;
}

/* Frame selection + temporal interpolation for one bone of one clip, in COMPONENT space.
 * Mirrors mrw_baked_sample_bone / mrw_bake_sample_xform. clip = (first_frame, frame_count, looping). */
Xform sampleClip(int bone, uvec4 clip, float t, float dur) {
    int firstFrame  = int(clip.x);
    int frameCount  = int(clip.y);
    bool looping    = clip.z != 0u;

    int   i0 = 0;
    float u  = 0.0;
    if (frameCount != 1 && dur != 0.0) {                 /* else static: i0=0, u=0                 */
        float tl   = looping ? modPos(t, dur) : clampf(t, 0.0, dur);
        float fpos = (tl / dur) * float(frameCount - 1);
        int   i    = int(floor(fpos));
        if (i > frameCount - 2) i = frameCount - 2;      /* never index past the last pair         */
        i0 = i;
        u  = fpos - float(i);
    }

    Xform a = decodeBone(bone, firstFrame + i0);
#if MARROW_TEMPORAL_INTERP
    if (u != 0.0) {
        Xform b = decodeBone(bone, firstFrame + i0 + 1);
        a.q = nlerp(a.q, b.q, u);                        /* nlerp q, lerp t/s, BEFORE compose       */
        a.t = mix(a.t, b.t, u);
        a.s = mix(a.s, b.s, u);
    }
#endif
    return a;
}

/* Resolve one influence bone to its component-space xform for this instance: clip A, then the
 * sanctioned ≤2-clip cross-fade - same hemisphere-corrected nlerp + linear t/s, weight w.
 * Endpoints early-out so neither clip pays dead fetches: the engine SHOULD canonicalize a completed
 * fade back to w=0 (B becomes the new A), so w∈{0,1} is only the brief transition boundary. */
Xform resolveBone(int bone, InstanceAnim inst) {
#if MARROW_CROSSFADE
    float w = inst.blend.x;
    if (w >= 1.0) return sampleClip(bone, inst.clipB, inst.times.y, inst.times.w);  /* B only */
    Xform xa = sampleClip(bone, inst.clipA, inst.times.x, inst.times.z);
    if (w > 0.0) {
        Xform xb = sampleClip(bone, inst.clipB, inst.times.y, inst.times.w);
        xa.q = nlerp(xa.q, xb.q, w);
        xa.t = mix(xa.t, xb.t, w);
        xa.s = mix(xa.s, xb.s, w);
    }
    return xa;
#else
    return sampleClip(bone, inst.clipA, inst.times.x, inst.times.z);
#endif
}

void main() {
    /* OpenGL GLSL: gl_InstanceID. (Vulkan GLSL uses gl_InstanceIndex and requires globals in a
     * UBO/push-constant block - compile with `-V`; see the README.) */
    InstanceAnim inst = instances[gl_InstanceID];

    vec3 posAccum = vec3(0.0);
    vec3 nrmAccum = vec3(0.0);
    vec3 tanAccum = vec3(0.0);

    for (int i = 0; i < MARROW_INFLUENCES; ++i) {
        float wt = aWeights[i];
        if (wt == 0.0) continue;
        Xform xf = resolveBone(aBones[i], inst);

        /* Compose A = s·R; the 3×4 palette matrix is M = [A | t]. Kept as R rows + (s,t) so
         * position uses A and normal/tangent use the rotation directly (no mat3). */
        vec3 r0, r1, r2;
        quatToRows(xf.q, r0, r1, r2);
        vec3 rp = vec3(dot(r0, aPos), dot(r1, aPos), dot(r2, aPos));            /* R·p              */

        /* Position: M·[p,1] = s·(R·p) + t. */
        posAccum += wt * (xf.s * rp + xf.t);

        /* Normal: cof(A) = cof(s·R) = s²·R, so weight the rotated normal by s². (MAY drop the
         * s² ONLY when every influencing bone shares one uniform scale - it cancels under normalize.) */
        vec3 rn = vec3(dot(r0, aNormal), dot(r1, aNormal), dot(r2, aNormal));   /* R·n              */
        nrmAccum += wt * (xf.s * xf.s) * rn;

        /* Tangent: a surface direction transforms by the forward map A = s·R, so weight by s
         * (NOT s²). Handedness (.w) is carried through unchanged. */
        vec3 rt = vec3(dot(r0, aTangent.xyz), dot(r1, aTangent.xyz), dot(r2, aTangent.xyz));
        tanAccum += wt * xf.s * rt;
    }

    /* model places the (skinned, root-motion-free) bind-space vertex into the world; root motion was
     * evaluated on the CPU and folded into inst.model. */
    vec4 worldPos = inst.model * vec4(posAccum, 1.0);
    gl_Position   = uViewProj * worldPos;

    /* normals/tangents: rotate by model's upper-left (assumed rigid/uniform here for the crowd tier;
     * a non-uniformly-scaled placement would use its inverse-transpose). */
    mat3 m3  = mat3(inst.model);
    vNormal  = normalize(m3 * nrmAccum);
    vTangent = vec4(normalize(m3 * tanAccum), aTangent.w);
}
