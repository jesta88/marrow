Blend, additive, and mask {#howto_blend}
=========================

marrow gives you the **2-input pose-combine primitives** — blend, make-additive, accumulate, and
per-joint masking — as pure functions over caller-owned memory. The *engine* owns the state
machine and the blend graph; marrow owns the math. The primitives operate on local poses
(`mrw_trs` arrays, as produced by clip sampling) *before* the hierarchy compose.

[TOC]

## The shared contract

Every pose-algebra function follows the same rules:

- Output goes through an out-param; the return is an `mrw_result`. Zero allocation.
- `joint_count == 0` is `MRW_OK`; `joint_count > joint_capacity` is `MRW_E_CAPACITY`.
- **Any non-finite input** (a pose component, a mask entry, a weight) ⇒ `MRW_E_RANGE` with no
  output written.
- Weights and mask entries are **clamped** to `[0, 1]` (not rejected).
- In-place output must be the *exact same* buffer as an input — partial overlap is undefined.

## Blend two poses

`mrw_pose_blend()` interpolates two poses: rotation by hemisphere-corrected nlerp, translation and
scale by lerp. The effective weight per joint is `clamp(w * (mask ? mask[j] : 1), 0, 1)`.

```c
/* a, b, out: joint_count mrw_trs each. mask is optional (NULL = uniform weight w). */
mrw_pose_blend(a, b, /*w=*/0.5f, /*mask=*/NULL, out, joint_count, joint_capacity);
```

`w = 0` yields `a`; `w = 1` yields `b`. `out` may alias `a` and/or `b`.

## Additive layers

An additive layer is built in two steps. First, turn an animated pose into a **delta** relative to
a reference (e.g. a "lean" clip relative to the idle base):

```c
mrw_pose_make_additive(pose, base, out_delta, joint_count, joint_capacity);
```

A delta is a *distinct semantic* — it is **not** a valid local pose, so never feed it to
`mrw_local_to_model()`. Then apply the delta on top of any base pose at a weight, with optional
mask:

```c
mrw_pose_accumulate(base, delta, /*w=*/1.0f, /*mask=*/NULL, out, joint_count, joint_capacity);
```

`accumulate(base, make_additive(pose, base), 1, NULL)` round-trips back to `pose` (translation
exact, scale exact for finite positive ratios, rotation up to sign).

## Per-joint masks

A mask is a `joint_count` array of `[0, 1]` weights — one per joint — that scales the effective
weight. This is how you confine a layer to part of the body: a mask that is `1` on the spine and
arms and `0` on the legs makes an "aim the upper body" additive leave the stride untouched. Pass it
as the `mask` argument to `mrw_pose_blend` or `mrw_pose_accumulate`; `NULL` means a uniform weight.

## The same thing, for a whole crowd

The batch path fuses these combines directly into the palette pipeline for `N` instances sharing
the skeleton and source clips:

- `mrw_batch_blend_clips_to_palette()` — per instance, blend `clipA@timesA[i]` with
  `clipB@timesB[i]` at `weights[i]`, optional shared mask.
- `mrw_batch_accumulate_to_palette()` — per instance, apply one shared additive `delta` on top of
  `base_clip@times[i]` at `weights[i]`, optional shared mask.

Both have `_f16` twins and follow the memory/threading model in @ref howto_crowd.

## Solving IK

For analytic constraints — two-bone limbs and aim/look-at — see the `mrw_ik_two_bone()` and
`mrw_ik_aim()` functions in the **Inverse kinematics** topic of the API reference. IK adjusts local
rotations to meet a model-space goal and is CPU-only (never batched).
