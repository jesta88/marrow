# marrow - Vulkan demo

A cross-platform (GLFW + modern Vulkan 1.4) demo that showcases the marrow animation runtime and
doubles as a live validation / stress harness. It is a **separate target** (like `tools/`): it may
use heavy dependencies (Vulkan, GLFW, volk) and is **never linked into the zero-dependency runtime**.

## Status (by phase)

- **Phase 0 - scaffold + Vulkan bring-up - done.** GLFW window; Vulkan 1.4 device gated on the modern
  feature set (dynamic rendering, synchronization2, timeline semaphores, buffer device address,
  descriptor indexing, `VK_EXT_shader_object`); shader objects + full dynamic state, no render
  passes / framebuffers / pipeline objects.
- **Phase 1 - procedural marrow assets - done.** A lean biped (14 joints) + idle/walk/run clips authored
  in code via `tools/authoring`, baked to a Tier-B palette with the `tools/bake` core, plus a procedural
  box-per-bone skinned mesh. No external files; the final `.mrw` self-validates through `mrw_blob_open`.
- **Phase 2 - Tier-B baked GPU crowd - done.** The baked palette stream uploaded to an RGBA16F texture
  (bindless), per-instance state in a buffer-device-address SSBO, one instanced draw - the reference
  Tier-B skinning shader (`shaders/crowd.vert`, adapted from `examples/skinning/marrow_skin.glsl`).
  Scales to tens of thousands of instances.
- **Phase 3 - Tier-A heroes + LOD promotion - done.** Foreground heroes run the CPU pose pipeline
  (sample → engine-side nlerp cross-fade → `mrw_local_to_model` → `mrw_model_to_palette`, **general
  cofactor normals**, `mrw_root_motion` walking) and upload palettes to a BDA buffer
  (`shaders/skin_tierA.vert`). (The standalone foreground heroes - and the **P** / `--promote` toggle
  that re-rendered them through the Tier-B baked path - were later folded into the unified LOD field
  (Phase 7), which promotes between the two tiers continuously, by camera distance.)
- **Phase 4 - stress + live validation - done.** `--validate` / `--validate-gpu` run five checks:
  single-clip exact-at-frame parity (sub-mm), between-frame and cross-fade gaps (measured, via the
  component-space oracle `mrw_bake_sample_xform`/`mrw_xform_nlerp`), the scalar/SSE2/AVX2 batch
  benchmark + SIMD parity, and a GPU-skin-vs-CPU-oracle SSBO readback.
- **Phase 5 - glTF asset path (UAL1) - done.** `gltf2marrow`'s parent-resolution + topological
  DFS is lifted into a shared helper (`tools/gltf2marrow/joint_order.{h,c}`) called by both the
  converter and the demo, so the demo's mesh loader (`assets_gltf.c`, cgltf) derives a *provably
  identical* `JOINTS_0`→marrow remap (joint names are only a checked cross-check). `--gltf PATH`
  loads a real character: the pre-baked `.mrw` (skeleton + clips + baked) plus the source glTF mesh,
  producing the same `ProcAssets` the procedural path does (so crowd/heroes/validate are unchanged).
  The `.mrw` is produced offline by `gltf2marrow` → `marrow-bake` - automated as the `demo_ual1`
  asset-prep target (see **glTF assets** below).
- **Phase 6 - benchmarking & profiling - done.** The demo now *measures* marrow honestly across CPU
  and GPU. A measurement backbone (`profiler.{c,h}`: one portable timer, named CPU scopes that
  **separate waits from work**, ring-buffer percentiles + EMA, a frame-time history ring) plus
  per-frame **GPU timestamp queries** in `vk_context`. A live in-window **HUD** (F1) shows FPS,
  CPU/GPU ms per stage - including the `PALETTE_GEN` (library batch kernel into cacheable RAM) vs
  `PALETTE_UPLOAD` (memcpy to the mapped SSBO) split - plus a frame-time graph (vendored
  `stb_easy_font`). A new **CPU-batch crowd tier** (`crowd_cpu.c`) drives the flagship
  `mrw_batch_clip_to_palette` *every frame* on a runtime-selectable backend (scalar/SSE2/AVX2),
  grouped per-clip into homogeneous batches - a faithful A/B against the GPU crowd, and the first
  time the demo exercises marrow's headline CPU claim in the render loop. That batch is also
  **jobified**: a small thread pool fans it across cores (the runtime owns no threads), with a
  bit-for-bit serial-vs-threaded parity gate (see **Jobified marrow** below). A headless **`--bench`**
  sweep (`{1k/4k/16k/65k} × {gpu, cpu-scalar/sse2/avx2}`) emits a human table + CSV/JSON for
  regression tracking - including a **rasterizer-discard vertex-skinning microbenchmark** that
  isolates the Tier-B GPU palette skinning from raster/fragment/overdraw (`gpu_skin − gpu_static`).
  The procedural box mesh now winds CCW-outward on every face so the crowd renders back-face-culled.
  Gated by `demo_validate` (CPU + SIMD + CPU-crowd-layout parity) and
  `demo_bench_smoke` (bench path + GPU-timestamp readback + output-schema check). See
  **Benchmarking & profiling** below.
- **Phase 7 - unified LOD field (the headline scene) - done.** The interactive demo is one entity
  field (`field.{c,h}`): every entity is classified each frame by distance to the camera into two
  *independent* LODs. The **animation tier** (radius `R_A`) sends near entities to CPU **Tier A**
  (`mrw_batch_blend_clips_to_palette`, an exact local-space clip cross-fade) and the rest to the baked
  GPU **Tier B** crowd; the **render LOD** (radius `R_mesh`) draws near entities as the full skinned
  mesh and the far tail as a cheap bone-line **skeleton** proxy (two line endpoints per bone instead of
  hundreds of mesh vertices, keeping the horizon off the vertex wall). Near and far run the *same*
  animation, differing only by tier - marrow's LOD-promotion contract made continuously visible: a CPU
  Tier-A foreground feeding a massive baked Tier-B crowd that recedes into a skeleton horizon. Tier B
  has no instance ceiling, so the field scales to 65 536. This folds the earlier standalone
  heroes + crowd-tier toggles (**P / T / B / J**, `--promote`, `--crowd-tier`) into one distance-driven
  field; the GPU-baked-vs-CPU-batch and jobified A/Bs now live in `--bench` / `--validate`. New live
  knobs - **[ ]** / **; '** for `R_A` / `R_mesh` and **K** for the whole-field skeleton view. Gated by
  `demo_field_smoke` (`--field-smoke`), which drives every field draw path - near Tier-A mesh, far
  Tier-B mesh band, far skeleton tail, and the global all-skeleton mode - and asserts each band
  actually populated.

## Build

Enable the demo when configuring (it is OFF by default so the zero-dep library/test build is
unaffected). The Vulkan SDK must be installed (provides headers + `glslc`); GLFW and volk are fetched
automatically via CMake `FetchContent` on the first configure.

```
cmake -S . -B build -G Ninja -DMRW_BUILD_DEMO=ON
cmake --build build --target marrow_demo
```

On Windows, bootstrap the MSVC environment first (`vcvars64.bat`). On Linux, a system Vulkan loader
(`libvulkan.so`) and X11/Wayland dev packages are required (GLFW pulls them).

## Run

```
build/demo/marrow_demo
```

The default scene is the **unified LOD field** (see Phase 7): a near CPU Tier-A foreground that
continues into a massive baked Tier-B crowd, the far tail collapsing to a bone-line skeleton horizon.

Controls: **WASD / Q / E** fly (Q / E down / up), hold **right mouse** to look, **Shift** to sprint,
**M** cycle the model (procedural biped + every baked model in the assets folder - see **glTF assets**
below), **− / =** halve / double the live entity count, **[ / ]** shrink / grow the Tier-A radius
`R_A` (how deep the near CPU band reaches), **; / '** shrink / grow the render-LOD radius `R_mesh`
(full skinned mesh inside, bone-line skeleton beyond), **K** toggle whole-field bone-line mode (draw
every entity as a skeleton), **R** reset the perf stats, **F1** toggle the HUD, **F2** HUD detail,
**Esc** to quit.

### Flags

| flag | effect |
|---|---|
| `--count N` | initial live entity count (default 8192 interactive, 65536 with `--screenshot`). Adjustable live with **− / =** (halve / double) in `[1, 65536]` - the field's far Tier-B tier has no 16k ceiling, so the count runs all the way to capacity; buffers are sized to that capacity once, so resizing only re-lays-out the grid (no reallocation) |
| `--lod-range R_A[,R_mesh]` | distance radii in world units (default `35,55`): `R_A` = the animation-tier radius (nearer → CPU Tier A, beyond → baked Tier B), optional `R_mesh` = the render-LOD radius (nearer → full mesh, beyond → bone-line skeleton). Live with **[ / ]** and **; / '** |
| `--skeleton` | start in whole-field bone-line mode - draw every entity as a skeleton (same as pressing **K**) |
| `--cam X,Y,Z[,yaw,pitch]` | override the camera (world units / radians) for a reproducible shot framing |
| `--no-hud` | start with the HUD hidden for clean captures (**F1** still toggles it back on interactively) |
| `--frames N` | render N frames and exit (headless) |
| `--screenshot PATH` | write a PNG of the last frame (swapchain readback). The capture is **deterministic** - fixed timestep + fixed camera + no input polling - so a given frame number is the same simulated time across runs. Defaults to 150 frames (deep enough to warm the GPU-timestamp readback and fill the frame-time graph) unless `--frames` overrides |
| `--video DIR` | dump every frame of a cinematic crowd fly-by as a PNG sequence (`DIR/frame_NNNNN.png`) - a high establishing overview swooping down into a low forward skim along the crowd, framed by a moving look-at target. Like `--screenshot` it is **deterministic** (fixed timestep + camera driven by the normalized clip time + no input), and it defaults the count to 65536 and the length to 600 frames (~10s at 1/60); `--count` / `--frames` override. The HUD is forced off. `DIR` must already exist. Encode the PNGs with e.g. `ffmpeg -framerate 60 -i DIR/frame_%05d.png -c:v libx264 -pix_fmt yuv420p flyby.mp4` |
| `--validate` | run the CPU parity gates (scalar vs SSE2/AVX2, jobified, f16) + the batch throughput microbench, print a report, exit (no window) |
| `--validate-gpu` | the above **plus** the GPU-skin-vs-CPU-oracle SSBO readback, for both f32 and f16 palettes (creates a hidden device) |
| `--bench` | headless hidden-window benchmark sweep (counts × tiers); print a table, exit |
| `--bench-out FILE` | with `--bench`, also write the sweep to `FILE` as `.csv` or `.json` (by extension) |
| `--smoke` | with `--bench`, run a tiny fixed config (the CTest variant) |
| `--selftest-assets` | generate the assets, print stats, exit (no Vulkan) |
| `--cycle-models` | headless smoke for the model switch: load each discovered model in turn (the **M**-key teardown/rebuild path) and exit - hidden window, no input needed |
| `--field-smoke` | headless smoke for the LOD field: drive every field draw path (near Tier-A mesh, far Tier-B mesh band, far skeleton tail, and the global all-skeleton mode), assert each band populated, exit - hidden window |
| `--gltf PATH` | **start** on a specific glTF instead of the procedural biped: read the mesh from `PATH` and the pre-baked `.mrw` beside it (`<stem>.mrw`). Optional - the demo auto-discovers baked models in its assets folder; this just picks the startup one |
| `--mrw PATH` | override the `.mrw` path used with `--gltf` (default: the glTF's sibling `<stem>.mrw`) |

## Benchmarking & profiling

The demo measures marrow honestly: it warms up before timing, reports percentiles, splits
CPU-core vs upload vs GPU, excludes waits from the timed window, and calls out the
visual-only-determinism + vsync/hidden-window caveats at each metric.

**Live HUD.** Press **F1** for an overlay with FPS, frame time, **CPU work ms** (sum of LOD partition +
palette + record + submit - GPU-backpressure waits are excluded), **GPU ms** (from timestamp queries;
`n/a` if the queue can't time-stamp), the **`palette gen` vs `upload`** split, and instance /
**triangle / line** / bone counts - triangles count the mesh-drawn instances, lines the bone-line
skeletons, while `bones` covers the whole field (every entity is animated regardless of how it's
drawn). A **LOD line** shows the live split: **near (Tier A, CPU)** vs **far (Tier B, baked GPU)**
instance counts, the far set's **mesh / skeleton** breakdown (full mesh inside `R_mesh`, bone-line tail
beyond), the current `R_A` / `R_mesh` radii, and the `lod` partition cost in ms - plus a
`K: all-skeleton` marker in whole-field bone-line mode, and a clamp warning if the near set exceeds the
Tier-A capacity (the overflow falls back to Tier B). It also shows **`ns/(inst·bone)`** - the near-band
batched blend (`palette gen`) normalized by the bone-instances it covered, the scale-invariant headline
throughput number (the same figure the `--validate` microbench prints). The tier reads `field` and the
backend is the near band's best SIMD kernel. **F2** expands a per-scope CPU (including `lod`) +
per-zone GPU breakdown - the GPU `crowd` / `heroes` / `skel` zones time the far mesh band, the near
mesh band, and all bone-line work separately. The window title carries a one-line summary even with the
overlay off.

**Live knobs.** **− / =** halve / double the entity count; **[ / ]** and **; / '** widen / narrow the
Tier-A radius `R_A` and the render-LOD radius `R_mesh`; **K** flips the whole field to bone lines - all
without leaving the loop, so you can watch the near/far split, `ns/(inst·bone)`, and the frame-time
graph respond in real time. Each change clears the rolling stats automatically (and **R** clears them
on demand) so the percentiles settle on the new configuration instead of averaging across the
transition.

**Two crowd tiers, A/B in the bench.** The headless **`--bench`** sweep drives the same crowd at the
same scale through both consumption tiers. The **GPU-baked** tier samples the baked palette texture on
the GPU (`palette gen` reads 0 - there is no CPU palette work). The **CPU-batch** tier runs
`mrw_batch_clip_to_palette` every frame, grouped per clip into homogeneous batches, into
64-byte-aligned cacheable RAM (timed `PALETTE_GEN`) and then `memcpy`s into the mapped SSBO (timed
`PALETTE_UPLOAD`); it sweeps scalar → SSE2 → AVX2, so `palette gen` moves (AVX2 < scalar) while the
upload stays flat. (Interactively, the unified LOD field shows both tiers *simultaneously* - a near
CPU Tier-A band continuing into a far baked Tier-B crowd - rather than toggling the whole crowd between
them.)

**Jobified marrow (multithreading).** marrow's runtime owns no threads - every batch/pose call is a pure
function over caller-owned buffers with no globals, no allocation, and no hidden cursor state (clip
sampling is stateless: it takes the time `t`, not an ozz-style `SamplingContext`). That makes it
trivially fannable across cores, and the CPU-batch tier shows it: a small persistent thread pool
(`jobs.{c,h}`, Win32 / pthreads) splits the crowd into contiguous instance ranges and runs one
`mrw_batch_clip_to_palette` per lane. The safe-fan-out contract is the whole point - the read-only
skeleton/clip views and the `mrw_dispatch` are **shared**; each lane writes a **disjoint** output
slice and uses its **own** scratch - so `palette gen` / `ns·(inst·bone)` drop by roughly the lane
count. The `demo_validate` gate exercises the fan-out and proves it **bit-for-bit identical** to the
serial batch (per-instance results
are partition-independent - the gate uses a prime instance count so the lane boundaries fall off the
8-wide SIMD tile grid, exercising partial tiles), for scalar and the host-best backend, and reports
the measured speedup - e.g. on a 16-lane host: scalar `~9 → ~1.3 ms` (~7×), AVX2 `~2.2 → ~0.36 ms`
(~6×); AVX2 + jobs is ~25× over scalar-serial. (Upload stays single-threaded - it is write-combined,
memory-bound, and timed separately.)

**Headless sweep.**

```
build/demo/marrow_demo --bench                       # table to stdout
build/demo/marrow_demo --bench --bench-out bench.csv  # + machine-readable CSV
build/demo/marrow_demo --bench --bench-out bench.json # + JSON (host/device/build metadata)
```

The sweep runs `{1000, 4000, 16384, 65536} × {gpu-baked, cpu-scalar, cpu-sse2, cpu-avx2}` (the CPU
tiers only up to the 16384 cap), with a fixed camera and an injected fixed `dt` so animation advances
reproducibly. Per config it reports CPU ms (mean/p95, waits excluded), GPU ms (mean/p95, timestamps),
the GPU `CROWD`-zone time, `PALETTE_GEN` + `PALETTE_UPLOAD` ms, and FPS (diagnostic - present-mode
dependent). For the **gpu-baked** tier it also runs a **vertex-skinning microbenchmark** - `gpu_skin`
(the real Tier-B skinning VS under rasterizer discard) and `gpu_static` (a matched no-palette
baseline); `gpu_skin − gpu_static` isolates the marrow palette skinning, reported scale-invariant as
`skin_ns/v` (ns per instance·vertex) - per-vertex (not per-bone) because the GPU re-skins per
vertex·influence, and a vertex-stage microbenchmark, not a frame cost. Sample (Release, RX
6950 XT, truncated):

```
tier       backend    count | cpu_mean  cpu_p95 | gpu_mean  gpu_p95 gpu_crwd | gpu_skin gpu_stat skin_ns/v |   gen_ms   upl_ms |      fps
gpu-baked  gpu        16384 |    0.216    0.266 |    3.235    3.257    2.792 |    2.805    1.852     0.135 |   0.0000   0.0000 |    239.8
cpu-batch  scalar     16384 |    9.657   10.586 |    3.031    3.154    2.633 |      n/a      n/a       n/a |   9.0822   0.4164 |     98.7
cpu-batch  AVX2       16384 |    2.770    3.222 |    3.092    3.126    2.647 |      n/a      n/a       n/a |   2.2951   0.3688 |    240.4
```

`gpu_skin ≈ gpu_crwd` here means the crowd draw is **vertex-bound** (skinning, not rasterization,
dominates - back-face culling on the closed box mesh keeps fragment/overdraw cheap); a raster-bound
draw would show `gpu_crwd ≫ gpu_skin`.

For meaningful kernel numbers build **Release** (`--preset x64-release-demo`); a Debug build runs the
batch kernels unoptimized. The bench uses a *hidden window* (a real swapchain, not a true offscreen
target), so FPS is bounded by the present mode - the surface-independent CPU-work and GPU-timestamp
columns are the metrics that matter.

**Automated gates** (run via `ctest`):

- `demo_validate` - CPU-only: exact-at-frame Tier-A/B parity, scalar/SSE2/AVX2 SIMD parity, the
  **CPU-crowd layout parity** (the batch AoS palette, indexed exactly as `skin_tierA.vert` does,
  skins identically to the per-instance `mrw_clip_to_palette` oracle), and the **concurrency parity**
  (the batch fanned across a real thread pool is bit-identical to the serial batch - the
  "concurrent calls sharing read-only assets + dispatch context" release gate).
- `demo_bench_smoke` - runs `--bench --smoke` to CSV + JSON and re-parses both (header, column count,
  finite numbers, all configs present), exercising the bench path, GPU-timestamp readback, and both
  tiers. Needs a Vulkan device + surface, like `--validate-gpu`.
- `demo_field_smoke` - runs `--field-smoke`: drives the unified LOD field through every draw path -
  near Tier-A mesh, far Tier-B mesh band, far bone-line skeleton tail, and the global all-skeleton
  mode - and asserts each band actually populated, so a regression that silently drops a draw fails
  instead of passing green. Needs a Vulkan device + surface.

## glTF assets (UAL1)

The glTF path consumes a `.mrw` produced **offline** by the two existing tools, exactly as a real
content pipeline would. The `demo_ual1` target automates the whole thing - take the committed
**UAL1** game character (`assets/UAL1_Standard_RM.glb`, a 65-bone Quaternius rig, CC0), convert, and
bake:

```
cmake --build build --target demo_ual1
```

This copies `UAL1_Standard_RM.glb` into `build/demo/assets/` (so the produced `.mrw` sits beside it),
then runs:

```
gltf2marrow build/demo/assets/UAL1_Standard_RM.glb -o build/demo/assets/UAL1_Standard_RM_rig.mrw --loop --anim Walk_Loop --anim Sprint_Loop
marrow-bake  build/demo/assets/UAL1_Standard_RM_rig.mrw -o build/demo/assets/UAL1_Standard_RM.mrw --mesh build/demo/assets/UAL1_Standard_RM.glb
```

Of UAL1's 43 animations only two locomotion loops are baked, in role order - `Walk_Loop` becomes
clip 0 (`walk`) and `Sprint_Loop` clip 1 (`run`), which is what the heroes cross-fade and the crowd
cycles. Run the demo on it (the demo derives the sibling `UAL1_Standard_RM.mrw` from the glTF path):

```
build/demo/marrow_demo --gltf build/demo/assets/UAL1_Standard_RM.glb
```

The same flags work - e.g. `--validate`, `--lod-range 40,70`, `--frames 30 --screenshot ual1.png`. The
field's Tier-B path requires the rig to be Tier-B-eligible (UAL1 is); an ineligible rig bakes a
Tier-A-only `.mrw`, which the field rejects (its far tier needs the `BAKED` section).

**Switching models in-app.** `--gltf` is only the *startup* model - the demo also scans its assets
folder (the directory next to the executable, e.g. `build/demo/assets/`) for every `.glb`/`.gltf`
that has a sibling baked `.mrw`, and **M** cycles through them plus the procedural biped at runtime
(it logs the scanned dir + model count at startup). **UAL1 is baked automatically as part of the
`marrow_demo` build** (`marrow_demo` depends on `demo_ual1`), so M toggles biped ↔ UAL1 out of the
box - no launch argument and no separate target build needed. Drop any other
`gltf2marrow`→`marrow-bake` output into that folder and it joins the cycle too. Switching rebuilds
every per-model GPU resource (mesh, both crowd tiers, heroes) for the new character while the camera,
entity count, and CPU-tier settings carry over. `--cycle-models` drives that path headlessly as a
smoke test.

> The `demo_ual1` target needs no network - the `.glb` is committed in-tree. The tool executables
> (`gltf2marrow`, `marrow-bake`) are built automatically when `-DMRW_BUILD_DEMO=ON`.

## Dependencies (demo target only)

- **Vulkan SDK** - headers + `glslc` (shaders compiled to SPIR-V at build time and embedded as C
  arrays, so the executable is path-independent).
- **GLFW 3.4** - window/surface/input (FetchContent).
- **volk** - Vulkan meta-loader for core 1.4 + extension entrypoints, including shader objects
  (FetchContent).
- **cgltf** (glTF path only) - the vendored `tools/third_party/cgltf` reader, linked via `mrw_cgltf`
  + `mrw_g2m`; used by `assets_gltf.c` to read the mesh and by the offline tools.

The shaders under `shaders/` are demo-specific; the normative Tier-B skinning reference lives in
`../examples/skinning/` (`marrow_skin.glsl`) and is what later phases adapt to Vulkan GLSL.
