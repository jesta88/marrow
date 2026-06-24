Animate a crowd {#howto_crowd}
===============

The batch path is marrow's reason to exist. One call animates `N` instances that **share a
skeleton and a clip**, each at its own time, and writes one contiguous palette buffer — vectorized
*across instances*, so the work stays cache-resident as the crowd grows. There are no per-instance
virtual calls and no per-instance allocations.

[TOC]

## 1. Detect a backend once

The SIMD backend is a small caller-owned, immutable POD value. Detect it once at startup and pass
it to every batch call. Buffer sizes never depend on it.

```c
mrw_dispatch disp;
mrw_dispatch_detect(&disp);   /* best of AVX2(+FMA) > SSE2 > scalar that the host supports */
```

`mrw_dispatch_scalar/sse2/avx2()` force a specific backend (the forced constructors return
`MRW_E_UNSUPPORTED` if the host can't run that ISA). The scalar backend is the bit-exact reference
the SIMD kernels are checked against.

## 2. Query sizes, bring your own memory

Ask how much scratch and output you need — the query returns both **size and alignment** — then
allocate it however you like. marrow never allocates.

```c
mrw_mem_req scratch_req, pal_req;
mrw_batch_clip_to_palette_requirements(joint_count, N, MRW_PALETTE_F32,
                                       &scratch_req, &pal_req);

void  *scratch  = my_alloc(scratch_req.align, scratch_req.size);   /* >=64-aligned */
float *palettes = my_alloc(pal_req.align,     pal_req.size);       /* >=16-aligned */
```

## 3. One call, N instances

```c
mrw_batch_clip_to_palette(&disp, &skel, &clip, times, N,
                          palettes, pal_req.size, scratch, scratch_req.size);
```

`times` is an array of `N` per-instance sample times. The output is the per-instance,
joint-contiguous AoS 3×4 palette: instance `i`, joint `j` occupies
`palettes[(i*joint_count + j)*12 .. +12)`, rows `row0, row1, row2`. Upload that straight to the GPU.

## Run it across your own threads

`mrw_batch_clip_to_palette()` is **side-effect-free**: it reads the shared, immutable skeleton and
clip views and writes only into the output and scratch you handed it. marrow owns no threads — *you*
schedule. The pattern is to shard the instances and give each worker its own scratch and its own
slice of the output:

```c
/* Per worker w covering instances [lo, hi): */
mrw_mem_req sreq, preq;
mrw_batch_clip_to_palette_requirements(joint_count, hi - lo, MRW_PALETTE_F32, &sreq, &preq);

mrw_batch_clip_to_palette(&disp, &skel, &clip,
                          times + lo, hi - lo,
                          palettes + (size_t)lo * joint_count * 12, preq.size,
                          per_thread_scratch[w], sreq.size);
```

Scratch must be **per worker** (it is mutated). The skeleton/clip views and the dispatch are
read-only and safe to share. `instance_count == 0` is a clean no-op.

## Halve the bandwidth with an f16 palette

Every batch entry point has an `_f16` twin that writes binary16 components instead of binary32 —
half the bytes to store, upload, and fetch, at a precision cost. Size it with
`out_format = MRW_PALETTE_F16` in the requirements query; everything else is identical.

```c
mrw_batch_clip_to_palette_f16(&disp, &skel, &clip, times, N,
                              (uint16_t *)palettes, pal_req.size, scratch, scratch_req.size);
```

## Beyond a single clip

To cross-fade two clips or apply an additive layer across the whole batch, use
`mrw_batch_blend_clips_to_palette()` and `mrw_batch_accumulate_to_palette()` — same memory model,
same threading story. See @ref howto_blend.

When even the CPU batch isn't enough for very distant crowds, bake the clip set into a GPU texture
tier: @ref concept_two_tiers.
