marrow {#mainpage}
======

**A pure-C11, zero-dependency, engine- and renderer-agnostic animation runtime for games.**

marrow turns skeletons and animation clips into skinning matrices. It is *batch-first*: one
call animates `N` instances that share a skeleton, so you can drive tens of thousands of
characters without tens of thousands of per-instance virtual calls. It never allocates, never
touches a graphics API, and never spawns a thread it doesn't own — it emits data and the format
specs to decode it, and leaves scheduling, GPU upload, and the animation state machine to you.

![marrow's unified LOD field: 65,536 animated characters](hero.jpg)

The entire public API is the single header `marrow.h`. All symbols are prefixed `mrw_`
(types/functions) or `MRW_` (macros/enums). The on-disk format is `.mrw` (magic `MRRW`).

## Get started

- @subpage tutorial_first — load a `.mrw`, animate one character, read its skinning palette.

## How-to guides

- @subpage howto_integrate — drop marrow into your build (and the one compile-flag rule that matters).
- @subpage howto_crowd — animate a crowd with the batch path, across your own threads.
- @subpage howto_blend — cross-fade clips, additive layers, and per-joint masks.
- @subpage howto_gltf — import a glTF 2.0 skin + animations into a `.mrw`.
- @subpage howto_baked_shader — decode the baked Tier-B palette in your vertex shader.

## Concepts

- @subpage concept_two_tiers — the CPU Tier A and the baked-GPU Tier B, and when to use each.
- @subpage concept_conventions — coordinate system, units, quaternion order, memory model.

## Reference

The **API reference** is generated from `marrow.h` — browse it under **Topics** (grouped by
area: loader, dispatch, pose pipeline, pose algebra, IK, batch, root motion) and
**Data Structures**. The pages below cover the rest:

- @subpage ref_errors — every `mrw_result` code and when it fires.
- @subpage ref_cmake — CMake build options.
- @subpage ref_format — the `.mrw` on-disk format at a glance.

## Why marrow

- **Zero runtime dependencies, zero allocation.** Pure C11, no libc beyond `<math.h>`. Every
  buffer is sized by a `*_requirements()` query that returns size *and* alignment; you bring the
  allocator. Console- and fixed-budget-friendly.
- **Flat C ABI.** `extern "C"`, compiles as both C11 and C++, out-params + result codes, no SIMD
  types in public structs. Trivial to bind from any language.
- **Batch-first and data-oriented.** The hot path fuses local → model → skinning and writes the
  canonical 3×4 palette directly, vectorized *across instances* (lane *i* = instance *i*).
- **Runtime SIMD dispatch.** Scalar, SSE2, and AVX2 (+FMA, +F16C) kernels; the backend is a small
  caller-owned POD value, detected once and passed in. Buffer sizes never depend on it.
- **Safe loader.** `.mrw` is a byte-defined little-endian format. The loader *validates, then
  views* — no struct overlay, no `mmap + cast`. Corrupt input fails cleanly.

## License

[MIT](https://github.com/jesta88/marrow/blob/master/LICENSE) — © 2026 Jérémie St-Amand.
