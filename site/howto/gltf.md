Import a glTF into a .mrw {#howto_gltf}
=========================

The marrow runtime only *consumes* assets. Producing them is the job of the offline tool
**`gltf2marrow`**, which imports a glTF 2.0 skin and its animations into a v0 `.mrw`. It is a
separate build target — it may use heavier libraries (it vendors `cgltf`) and is never linked into
the zero-dependency runtime.

[TOC]

## Build the tool

The offline tools build by default (`MRW_BUILD_TOOLS` is `ON`):

```sh
cmake -S . -B out/build -G Ninja
cmake --build out/build --target gltf2marrow
```

## Convert an asset

```sh
gltf2marrow character.gltf -o character.mrw
```

The importer:

- folds the non-joint node chain (the Armature wrapper and any intermediate nodes) onto marrow's
  joint→joint local transforms;
- **resamples** each animation onto a dense, duration-preserving clip (handling STEP, LINEAR/slerp,
  and CUBICSPLINE interpolation in the source);
- **self-validates** the result through `mrw_blob_open()` before writing — a file `gltf2marrow`
  emits is a file the runtime will accept.

The output is a v0 `.mrw` containing the skeleton and one clip per source animation. Load it exactly
as in @ref tutorial_first.

## Baking a crowd tier

To produce the baked GPU "Tier B" texture section from a `.mrw` clip set, use the companion tool
**`marrow-bake`**:

```sh
marrow-bake character.mrw -o character_baked.mrw [--bake-fps N] [--mesh character.gltf]
```

It samples Tier A at the bake rate, polar-decomposes each bone's skinning transform to
`Q + T + uniform-scale`, measures the reconstruction residual against probe points, and **rejects
rigs that cannot be represented** (the whole rig, not silently per-bone) — those stay perfectly
valid Tier-A assets; there is no lossy full-matrix fallback. See @ref concept_two_tiers for what
the baked tier is and when to use it, and @ref howto_baked_shader to decode it on the GPU.
