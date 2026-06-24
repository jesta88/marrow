The two-tier model {#concept_two_tiers}
==================

marrow is one skeletal pipeline with two consumption tiers, related as a fidelity LOD. They share
the same skeleton/clip formats, decompressor, sampler, and math.

```
                ┌──────────── shared core (C, SoA SIMD, no alloc) ────────────┐
  .mrw blob ──► │ skeleton │ clip decode │ sample │ blend/additive/mask │ IK │ │
                └──────────────┬──────────────────────────────────┬───────────┘
                               │                                  │ (offline tool only)
                   Tier A: CPU runtime pose           Tier B: baked per-bone palettes
                   full control — blend trees,         fixed clip set, fixed rate,
                   IK, partial blends, root motion     ≤2-clip cross-fade in shader
                               │                                  │
                               ▼                                  ▼
                  canonical 3×4 skinning palette  ⇄  shader decode (same layout)
```

[TOC]

## Tier A — the CPU runtime

Tier A samples clips, blends/masks/adds poses, solves IK, runs the hierarchy, and writes the
skinning palette — full per-frame control, engine-driven. This *is* the marrow runtime library.
Use it for anything that needs exact, per-frame, local-space control: the player, NPCs near the
camera, anything gameplay-critical. The batch path (@ref howto_crowd) makes Tier A scale to large
crowds on its own.

## Tier B — the baked GPU crowd tier

Tier B bakes a fixed clip set offline into per-bone `Q + T + scale` textures whose memory scales
with `bones × frames`, **not vertices**. The runtime *consumes and validates* the baked blob; a
reference vertex shader decodes it on the GPU (@ref howto_baked_shader). Use it for very distant
crowds, where per-frame CPU control is wasted.

## The honest relationship

Tier B is **exact at baked frames** (within half-float quantization) and **approximate between
them**: it interpolates already-composed transforms — a chord across the motion, not the arc — so
it is a frozen cache of discrete Tier-A palettes with approximate temporal interpolation.

This is the guardrail that keeps marrow small: **Tier B never grows GPU hierarchy evaluation, masks,
IK, or runtime graphs.** Anything that needs those promotes the entity back to Tier A. There is one
animation engine (Tier A) and one frozen cache of it (Tier B) — not two engines.

## When to promote

| Situation | Tier |
|---|---|
| Player, gameplay-critical, near camera | A |
| Needs IK, masks, additive layers, blend trees, root motion | A |
| Exact local-space blending matters | A |
| Distant crowd, fixed clip set, ≤2-clip cross-fade | B |
| Bone-count-bound rather than vertex-bound memory budget | B |

Promotion is a per-entity, per-frame decision the engine makes by distance/importance; both tiers
produce the *same* canonical 3×4 palette layout, so the handoff is seamless. (A rig that cannot be
faithfully represented as `Q + T + uniform-scale` is rejected by the baker entirely and simply stays
a Tier-A asset — there is no lossy fallback.)
