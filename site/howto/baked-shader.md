Decode the baked Tier-B palette in a shader {#howto_baked_shader}
===========================================

Tier B is a **frozen, precomputed cache** of Tier A: a fixed clip set baked at a fixed rate into a
per-bone `Q + T + scale` texture, decoded in a vertex shader for distant crowds. This guide is
about decoding that texture. It is deliberately narrow — Tier B decodes baked transforms and
nothing more. Anything richer (masks, additive layers, IK, blend graphs, hierarchy evaluation)
**promotes the entity back to Tier A**; marrow never grows a second animation engine on the GPU.
See @ref concept_two_tiers for that boundary.

[TOC]

## Reference shaders

marrow ships reference vertex shaders that consume the `.mrw` BAKED section and skin a mesh:

- `examples/skinning/marrow_skin.glsl` — canonical reference (GLSL 450).
- `examples/skinning/marrow_skin.hlsl` — HLSL SM5 port (consoles / D3D).

These are **reference material only**: not part of the runtime, no GPU code in the library, not
built by CMake (there is no GPU toolchain in CI). The runtime stays consume-only. The **normative
contract** — texture upload/packing, the per-instance I/O, and the decode/compose/skin math, each
step cross-referenced to the gated CPU code it transcribes — is documented inline in
`marrow_skin.glsl`. Start there.

## What the shader does (and doesn't)

- Sample the palette **only** via integer fetch (`texelFetch` / `Texture2D.Load`) — **no hardware
  filtering**. All interpolation is hemisphere-corrected nlerp in component space, matching the CPU.
- Animation state is **per-instance** (one entry per drawn entity).
- The default path matches the runtime `mrw_baked_sample_bone()` exactly; the `MARROW_*` `#define`
  knobs select progressively cheaper, *approximate* distant-crowd LODs.
- The shader does **no** inverse-bind multiply (it is pre-applied into the bake), **no** root motion
  (the engine applies the clip `root_track` on the CPU), and **no** hierarchy / IK / masks.

## The honest fidelity story

Tier B is **exact at baked frames** (within half-float quantization) and **approximate between
them** — it interpolates already-composed transforms (a chord, not the arc), so it is a frozen
cache of discrete Tier-A palettes with approximate temporal interpolation. That is the right
trade for distant crowds. Promote to Tier A when exact local-space blending matters — near camera,
or gameplay-critical. See @ref concept_two_tiers.

## Optional syntax check

Not required, not in CI. The GLSL targets OpenGL GLSL 450:

```sh
glslangValidator -S vert examples/skinning/marrow_skin.glsl
dxc -T vs_5_0 -E VSMain examples/skinning/marrow_skin.hlsl
```
