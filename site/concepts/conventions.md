Conventions and memory model {#concept_conventions}
============================

These are the assumptions baked into every marrow function. They never change at runtime.

[TOC]

## Coordinate system and math

| | |
|---|---|
| Handedness | right-handed, **+Y up** |
| Units | meters, seconds, radians |
| Vectors | column vectors; a point transforms as `p' = M · [p, 1]` |
| Quaternions | unit, component order **(x, y, z, w)**, Hamilton product |
| Local compose order | `local = T · R · S` (translate · rotate · scale) |
| Numbers | little-endian, IEEE-754 |

Transforms are stored as **3×4 affine** matrices (rows `row0, row1, row2`; the 3×3 basis plus a
translation column). A `mrw_trs` is a local transform with non-uniform scale; a `mrw_xform` is a
rigid transform with one uniform scale (used by the baked encoding and root-motion deltas).

## The ABI contract

marrow is built to be bound from any language:

- Every output goes through an **out-param** plus an `mrw_result` return code.
- Functions never return a nontrivial struct by value.
- Counts are unsigned and fixed-width.
- **No compiler SIMD types** appear in public structs.
- The header compiles as both **C11 and C++**.

The pure math primitives (the **Math core** topic) are the exception: they are total and infallible,
so they take out-params but return `void` (or a trivially-copyable scalar) rather than `mrw_result`.

## Memory model: bring your own allocator

The runtime **never allocates** and beyond `<math.h>` uses no libc. Every buffer you must provide is
sized by a `*_requirements()` query that returns **both size and alignment**:

```c
mrw_mem_req scratch_req, pal_req;
mrw_batch_clip_to_palette_requirements(joint_count, N, MRW_PALETTE_F32, &scratch_req, &pal_req);
/* allocate scratch_req.size at scratch_req.align, etc. — however you like */
```

Buffer sizes are **dispatch-independent**: the chosen SIMD backend never changes them. Typical
alignment requirements are 64 bytes (cache line) for batch scratch and 16 bytes for palette output;
the single-character path's scratch/output buffers want ≥16. The blob you hand `mrw_blob_open()`
must be ≥64-byte aligned.

## Threading

marrow owns **no threads**. The batch entry points are side-effect-free: they read shared,
immutable views and write only the output and scratch you pass in. You schedule them across your own
workers — shared read-only views, per-thread scratch. See @ref howto_crowd.

## Determinism

Determinism is **visual-only**. FMA contraction and reordered reductions are deliberately allowed,
so results are *not* bit-identical across machines or backends. The scalar backend is the bit-exact
reference; SSE2 and AVX2 match it within a small, visual-only tolerance. If you need cross-machine
bit-determinism (e.g. lockstep networking), do not rely on the palette bytes matching.

## Identities

Skeletons and clips carry a 128-bit content fingerprint (FNV-1a-128 over a layout-independent
canonical stream). marrow uses it to check that a clip and skeleton (or a baked section and its
skeleton) actually belong together — a mismatch is `MRW_E_INCOMPATIBLE` rather than silent garbage.
