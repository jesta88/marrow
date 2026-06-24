#version 450
/* Differential baseline for the Tier-B vertex-skinning microbenchmark (BENCH-ONLY, rasterizer
 * discard on). It pays EXACTLY the per-vertex costs crowd.vert pays - same vertex-attribute fetch,
 * same full InstanceAnim load, the same 4-influence loop + weight blend + accumulation, the same
 * model/clip-space transform - but does NO palette texelFetch / Q+T+s decode / nlerp. So
 *   gpu_skin_ms - gpu_static_ms  ==  the marrow Tier-B palette sample+decode+blend, isolated.
 * Mirror crowd.vert's interface byte-for-byte (inputs, push constant, instance buffer) so the
 * subtraction cancels everything except the palette work. */
#extension GL_EXT_buffer_reference2     : require
#extension GL_EXT_scalar_block_layout    : require

#define MARROW_INFLUENCES 4

struct InstanceAnim {
    mat4  model;
    uvec4 clipA;
    uvec4 clipB;
    vec4  times;
    vec4  blend;
};
layout(buffer_reference, scalar) readonly buffer InstanceRef { InstanceAnim instances[]; };

layout(push_constant, scalar) uniform PC {
    mat4        viewProj;
    InstanceRef ref;
    uint        texelsPerBone;
    float       benchEps;        /* folds the result into gl_Position (anti-DCE); raster is off */
} pc;

layout(location = 0) in vec3  aPos;
layout(location = 1) in vec3  aNormal;
layout(location = 2) in vec4  aTangent;   /* declared to match crowd.vert's input layout (unused) */
layout(location = 3) in uvec4 aBones;
layout(location = 4) in vec4  aWeights;

void main() {
    InstanceAnim inst = pc.ref.instances[gl_InstanceIndex];

    vec3 posAccum = vec3(0.0), nrmAccum = vec3(0.0);
    float touch = 0.0;
    for (int i = 0; i < MARROW_INFLUENCES; ++i) {
        float wt = aWeights[i];
        if (wt == 0.0) continue;
        touch += float(aBones[i]);     /* keep aBones live (it indexes the palette in crowd.vert) */
        posAccum += wt * aPos;         /* stand-in transform: identity, no palette fetch */
        nrmAccum += wt * aNormal;
    }

    /* Touch the same InstanceAnim fields crowd.vert's resolveBone() reads, so the load cost cancels. */
    touch += inst.blend.x + inst.times.x + inst.times.y + inst.times.z + inst.times.w
           + float(inst.clipA.x + inst.clipA.y + inst.clipA.z + inst.clipA.w)
           + float(inst.clipB.x + inst.clipB.y + inst.clipB.z + inst.clipB.w);

    vec4 worldPos = inst.model * vec4(posAccum, 1.0);
    gl_Position = pc.viewProj * worldPos;
    gl_Position += pc.benchEps * vec4(nrmAccum.x + touch, nrmAccum.y, nrmAccum.z, 0.0);
}
