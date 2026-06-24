# marrow - reference Tier-B GPU skinning shaders

Reference vertex shaders that consume the baked per-bone palette stream (the `.mrw` BAKED section)
and skin a mesh on the GPU - the Tier-B "crowd" consumer.

- [`marrow_skin.glsl`](marrow_skin.glsl) - canonical reference (GLSL 450).
- [`marrow_skin.hlsl`](marrow_skin.hlsl) - HLSL SM5 port (consoles / D3D).

**Reference material only.** These are *not* part of the zero-dependency `marrow` runtime, add **no** GPU
code to the library, and are **not** built by CMake (no GPU toolchain in CI). The runtime stays
consume-only.

The **normative contract** - the texture upload/packing contract, the per-instance I/O contract, the
sampling/decode/compose/skin math (each step cross-referenced to the gated CPU code it transcribes), the
GPU fetch budget + LOD knobs, the `.mrw` BAKED wire layout, and the verification recipe - is documented
inline in [`marrow_skin.glsl`](marrow_skin.glsl). Start there.

Quick orientation:

- Sample the palette **only** via integer fetch (`texelFetch` / `Texture2D.Load`), **no hardware
  filtering** - all interpolation is hemisphere-corrected nlerp in component space.
- Animation state is **per-instance** (one entry per drawn entity) - marrow is built for tens of
  thousands of entities.
- The default path matches the runtime `mrw_baked_sample_bone` exactly; the
  `MARROW_*` `#define` knobs select **approximate** distant-crowd LODs.
- The shader does **no** inverse-bind multiply (pre-applied), **no** root motion (engine applies the CLIP
  `root_track` on the CPU), and **no** hierarchy/IK/masks - Tier B is a frozen cache of Tier A.

Optional syntax check (not required, not in CI). The GLSL is **OpenGL GLSL 450** (`gl_InstanceID`,
default-block uniforms) - `-S vert` names the stage for the `.glsl` extension, no `-V`:

```
glslangValidator -S vert examples/skinning/marrow_skin.glsl
dxc -T vs_5_0 -E VSMain examples/skinning/marrow_skin.hlsl
```

(A Vulkan GLSL target uses `gl_InstanceIndex`, moves the globals into a UBO/push-constant block, and
compiles with `-V`.)
