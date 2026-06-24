Your first animated character {#tutorial_first}
=============================

This walks the whole Tier-A path end to end: build the library, load a `.mrw` asset, sample a
clip, and read out the skinning palette you would upload to the GPU. By the end you will have run
a real program — `examples/quickstart.c`, which is compiled and run by the test suite, so the code
below is exactly the code that ships.

[TOC]

## 1. Build the library

marrow builds with CMake (Ninja recommended). The runtime is just the C sources under `src/`
plus `marrow.h` — no third-party dependencies.

```sh
cmake -S . -B out/build -G Ninja
cmake --build out/build
```

This also builds the examples (`MRW_BUILD_EXAMPLES` is `ON` by default). To drop marrow into an
existing build instead, see @ref howto_integrate.

## 2. Load and validate a blob

A `.mrw` file is loaded by *validating, then viewing*: `mrw_blob_open()` runs the full structural
validation pass once, and on `MRW_OK` every accessor afterward is a checked locator into the
immutable bytes. There is no `mmap + cast`. The one requirement on you: the buffer must be at
least **64-byte aligned**.

```c
mrw_blob blob;
if (mrw_blob_open(data, size, &blob) != MRW_OK) {
    /* reject — the file is corrupt or incompatible */
}
```

## 3. Take the skeleton and a clip

A `.mrw` holds at most one skeleton, so there is a direct accessor for it. To find a clip, scan
the section table by type rather than hard-coding an index — asset layouts change.

```c
mrw_skeleton_view skel;
mrw_blob_skeleton(&blob, &skel);

mrw_clip_view clip;
for (uint32_t i = 0; i < blob.section_count; ++i) {
    uint32_t type = 0;
    if (mrw_blob_section_type(&blob, i, &type) == MRW_OK && type == MRW_SECTION_CLIP) {
        mrw_clip_view_at(&blob, i, &clip);
        break;
    }
}
```

## 4. Produce the skinning palette

`mrw_clip_to_palette()` is the fused convenience entry point: it samples the clip at time `t`,
composes the hierarchy, and applies the inverse-bind, writing one canonical **3×4** matrix per
joint. You provide two caller-owned, ≥16-byte-aligned buffers of `joint_count * 12` floats — one
scratch, one output. marrow does not allocate.

```c
float scratch[MAX_JOINTS * 12];   /* >=16-aligned */
float palette[MAX_JOINTS * 12];   /* >=16-aligned */

mrw_result r = mrw_clip_to_palette(&skel, &clip, t, scratch, palette, MAX_JOINTS);
/* palette[(j*12) .. +12) is joint j's 3x4 matrix: rows row0, row1, row2. */
```

A dense clip's duration is `(sample_count - 1) / fps`; sampling at `t` between frames interpolates.
Upload `palette` to the GPU as your skinning matrices, or feed it to your own skinning code.

## 5. The complete program

Everything above, assembled into a runnable program with error checking and a portable aligned
allocator (note: `aligned_alloc` is C11; MSVC spells it `_aligned_malloc`):

@include quickstart.c

Run it against the bundled fixture:

```sh
./out/build/examples/mrw_quickstart fixtures/rig.mrw
```

```
skeleton: 3 joints
clip: 2 samples @ 1.0 fps (1.000s); sampled t=0.500
joint 0 palette row0 = [ 1.000  0.000  0.000 |  0.000]
```

## Where to next

- **A crowd, not one character?** That is what marrow is built for — see @ref howto_crowd.
- **Blending and additive layers?** @ref howto_blend.
- **The conventions** (axes, units, quaternion order, who owns memory): @ref concept_conventions.
