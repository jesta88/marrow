// marrow - reference Tier-B GPU skinning vertex shader (HLSL SM5 port of marrow_skin.glsl).
//
// Same algorithm as the GLSL reference; see marrow_skin.glsl for the full
// rationale, the texture-upload contract, and the verification recipe. Consumes the baked palette
// stream (.mrw BAKED, encoding 1:
// Q+T+uniform-scale, 2 RGBA16F texels/bone, frame-major) and is a faithful transcription of the
// gated CPU path (src/marrow_pose.c mrw_baked_sample_bone, src/marrow_math.c, tools/bake/mrw_bake.c).
//
// GUARDRAIL: Tier B is a FROZEN cache of Tier A - no hierarchy/IK/masks/blend-graph on
// the GPU, no inverse-bind multiply (pre-applied in M), no root motion in the palette (the
// engine evaluates the CLIP root_track on the CPU and folds it into the per-instance model matrix).
//
// Reference material - NOT part of the zero-dependency runtime and NOT built by CMake.

// ----------------------------------------------------------------- feature knobs (LOD)
// DEFAULT path is the exact runtime reference math (matches mrw_baked_sample_bone). Reduced settings are
// *approximate* distant-crowd LODs, explicitly NOT equal to the runtime.
#define MARROW_INFLUENCES        4   // bones/vertex (reference 4). Fewer = approximate.
#define MARROW_TEMPORAL_INTERP   1   // 1 = nlerp/lerp (reference). 0 = hold previous (floor) frame, APPROXIMATE.
#define MARROW_CROSSFADE         1   // 1 = sanctioned ≤2-clip cross-fade. 0 = single clip.

// GPU fetch budget (worst case): influences × 2 texels × 2 frames × 2 clips. Reference = 32/vertex.

// ----------------------------------------------------------------- bindings
// Per-INSTANCE animation state (crowd-scale), one entry per drawn entity, indexed by SV_InstanceID.
// The StructuredBuffer transport is ENGINE POLICY (an engine may use per-instance vertex attributes
// instead); either feeds the same pure math functions.
struct InstanceAnim {
    float4x4 model;  // bind/model space -> WORLD (engine folds CPU-evaluated root motion in here)
    uint4    clipA;  // (first_frame, frame_count, looping(0/1), _pad)  - baked clip-table entry A
    uint4    clipB;  // (first_frame, frame_count, looping(0/1), _pad)  - entry B (cross-fade)
    float4   times;  // (tA, tB, source_duration_A, source_duration_B)
    float4   blend;  // (w, _, _, _)  - cross-fade weight w ∈ [0,1]; w==0 ⇒ clip A only
};
StructuredBuffer<InstanceAnim> gInstances : register(t1);

// Baked palette packed into an RGBA16F 2D texture. Sampled ONLY via .Load (integer, no sampler ⇒ no
// filtering): the decode needs hemisphere-corrected nlerp in COMPONENT space, and bilinear would lerp
// quaternions wrongly and bleed across bone/frame boundaries.
Texture2D<float4> gPalette : register(t0);

// Per-fetch addressing inputs. This reference implements the RECOMMENDED 2D-block packing
// (each skeleton block = frame_stride_texels wide × total_frames rows; a frame is one texture row,
// a bone is texels_per_bone texels across), so it addresses by (origin + bone·texels_per_bone + c,
// origin + F) and needs only texels_per_bone + the block origin. frame_stride_texels is an
// UPLOAD/PACKING parameter (block row width), not a per-fetch input; a linear/row-wrapping buffer
// would use the linear address with explicit stride, and an array atlas would add a layer.
cbuffer Globals : register(b0) {
    float4x4 gViewProj;          // world -> clip
    int      gTexelsPerBone;     // = 2 (q texel + t/s texel)
    int2     gPaletteOrigin;     // atlas origin (x,y) of this skeleton block
};

struct VSInput {
    float3 pos     : POSITION;   // bind-pose position
    float3 normal  : NORMAL;     // bind-pose normal
    float4 tangent : TANGENT;    // bind-pose tangent; .w = bitangent handedness (±1)
    int4   bones   : BLENDINDICES;
    float4 weights : BLENDWEIGHT; // ASSUMED pre-normalized (Σ=1, glTF); not renormalized here
};

struct VSOutput {
    float4 clipPos : SV_Position;
    float3 normal  : NORMAL;     // world-space
    float4 tangent : TANGENT;    // world-space .xyz + preserved handedness .w
};

// component-space transform of one bone at one phase (pre-compose)
struct Xform { float4 q; float3 t; float s; };

// ----------------------------------------------------------------- scalar helpers (explicit)

float clampf(float t, float lo, float hi) { return t < lo ? lo : (t > hi ? hi : t); }

// mrw_mod_pos. HLSL fmod is TRUNCATED (matches C fmodf), then BOTH guards: the +d can round to
// exactly d in float, so the trailing subtract re-applies the outer modulo.
float modPos(float t, float d) {
    float m = fmod(t, d);
    if (m < 0.0) m += d;
    if (m >= d) m -= d;
    return m;
}

// Quat (x,y,z,w) -> rotation ROWS (column-vector v'=R·v). Explicit rows avoid float3x3/mul
// row-vs-column ambiguity. Mirrors mrw_quat_to_mat3.
void quatToRows(float4 q, out float3 r0, out float3 r1, out float3 r2) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;
    r0 = float3(1.0 - 2.0*(yy+zz),  2.0*(xy-wz),        2.0*(xz+wy));
    r1 = float3(2.0*(xy+wz),        1.0 - 2.0*(xx+zz),  2.0*(yz-wx));
    r2 = float3(2.0*(xz-wy),        2.0*(yz+wx),        1.0 - 2.0*(xx+yy));
}

// full-precision renormalize: 1/sqrt(|q|²) (not rsqrt - its approximation would drift from the oracle).
float4 renorm(float4 q) {
    float n2 = dot(q, q);
    return q * (n2 > 0.0 ? 1.0 / sqrt(n2) : 0.0);
}

// Hemisphere-corrected nlerp; flip on dot < 0.0 (value compare, not the raw sign bit).
float4 nlerp(float4 a, float4 b, float u) {
    if (dot(a, b) < 0.0) b = -b;
    return renorm(lerp(a, b, u));   // lerp(a,b,u) = a + u·(b−a)
}

// Decode of one bone at absolute baked frame F: two integer loads, renormalize q.
Xform decodeBone(int bone, int F) {
    int x0 = gPaletteOrigin.x + bone * gTexelsPerBone;   // b·texels_per_bone (+c)
    int y  = gPaletteOrigin.y + F;                       //       absolute frame row
    float4 t0 = gPalette.Load(int3(x0,     y, 0));       // c=0: q
    float4 t1 = gPalette.Load(int3(x0 + 1, y, 0));       // c=1: (t, s)
    Xform o;
    o.q = renorm(t0);
    o.t = t1.xyz;
    o.s = t1.w;
    return o;
}

// Frame select + temporal interp for one bone of one clip, in COMPONENT space.
Xform sampleClip(int bone, uint4 clip, float t, float dur) {
    int  firstFrame = (int)clip.x;
    int  frameCount = (int)clip.y;
    bool looping    = clip.z != 0u;

    int   i0 = 0;
    float u  = 0.0;
    if (frameCount != 1 && dur != 0.0) {                 // else static: i0=0, u=0
        float tl   = looping ? modPos(t, dur) : clampf(t, 0.0, dur);
        float fpos = (tl / dur) * (float)(frameCount - 1);
        int   i    = (int)floor(fpos);
        if (i > frameCount - 2) i = frameCount - 2;
        i0 = i;
        u  = fpos - (float)i;
    }

    Xform a = decodeBone(bone, firstFrame + i0);
#if MARROW_TEMPORAL_INTERP
    if (u != 0.0) {
        Xform b = decodeBone(bone, firstFrame + i0 + 1);
        a.q = nlerp(a.q, b.q, u);                        // nlerp q, lerp t/s, BEFORE compose
        a.t = lerp(a.t, b.t, u);
        a.s = lerp(a.s, b.s, u);
    }
#endif
    return a;
}

// clip A, then the sanctioned ≤2-clip cross-fade. Endpoints early-out so neither clip pays
// dead fetches; the engine SHOULD canonicalize a completed fade to w=0 (B becomes the new A).
Xform resolveBone(int bone, InstanceAnim inst) {
#if MARROW_CROSSFADE
    float w = inst.blend.x;
    if (w >= 1.0) return sampleClip(bone, inst.clipB, inst.times.y, inst.times.w);  // B only
    Xform xa = sampleClip(bone, inst.clipA, inst.times.x, inst.times.z);
    if (w > 0.0) {
        Xform xb = sampleClip(bone, inst.clipB, inst.times.y, inst.times.w);
        xa.q = nlerp(xa.q, xb.q, w);
        xa.t = lerp(xa.t, xb.t, w);
        xa.s = lerp(xa.s, xb.s, w);
    }
    return xa;
#else
    return sampleClip(bone, inst.clipA, inst.times.x, inst.times.z);
#endif
}

VSOutput VSMain(VSInput v, uint instanceID : SV_InstanceID) {
    InstanceAnim inst = gInstances[instanceID];

    float3 posAccum = float3(0.0, 0.0, 0.0);
    float3 nrmAccum = float3(0.0, 0.0, 0.0);
    float3 tanAccum = float3(0.0, 0.0, 0.0);

    [unroll]
    for (int i = 0; i < MARROW_INFLUENCES; ++i) {
        float wt = v.weights[i];
        if (wt == 0.0) continue;
        Xform xf = resolveBone(v.bones[i], inst);

        // Compose A = s·R; M = [A | t]. Kept as rows + (s,t): position uses A, normal/tangent
        // use the rotation directly (no float3x3, avoids the row/column ambiguity).
        float3 r0, r1, r2;
        quatToRows(xf.q, r0, r1, r2);
        float3 rp = float3(dot(r0, v.pos), dot(r1, v.pos), dot(r2, v.pos));        // R·p

        // Position: M·[p,1] = s·(R·p) + t.
        posAccum += wt * (xf.s * rp + xf.t);

        // Normal: cof(s·R) = s²·R ⇒ weight rotated normal by s². (MAY drop s² ONLY if all
        // influencing bones share one uniform scale - it cancels under normalize.)
        float3 rn = float3(dot(r0, v.normal), dot(r1, v.normal), dot(r2, v.normal)); // R·n
        nrmAccum += wt * (xf.s * xf.s) * rn;

        // Tangent: surface direction transforms by A = s·R ⇒ weight by s (not s²); .w preserved.
        float3 rt = float3(dot(r0, v.tangent.xyz), dot(r1, v.tangent.xyz), dot(r2, v.tangent.xyz));
        tanAccum += wt * xf.s * rt;
    }

    // model places the skinned (root-motion-free) bind vertex into the world; root motion was
    // evaluated on the CPU and folded into inst.model. Matrix application uses the engine's
    // own convention (here mul(M, v), column-vector); the marrow-specific skinning above is
    // convention-explicit. Adjust to your engine's matrix-major-ness as needed.
    float4 worldPos = mul(inst.model, float4(posAccum, 1.0));
    VSOutput o;
    o.clipPos = mul(gViewProj, worldPos);

    float3x3 m3 = (float3x3)inst.model;   // rigid/uniform crowd placement; non-uniform would use inv-transpose
    o.normal  = normalize(mul(m3, nrmAccum));
    o.tangent = float4(normalize(mul(m3, tanAccum)), v.tangent.w);
    return o;
}
