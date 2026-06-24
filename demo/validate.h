/* Live validation + stress harness - the "validate the library" payoff.
 *
 * (1) single-clip exact-at-frame parity: CPU mrw_clip_to_palette vs baked mrw_baked_sample_bone
 *     at a baked frame (sub-mm half-float).
 * (2) between-frame error: the same off-frame - the chord-vs-arc gap, reported (not asserted == 0).
 * (3) cross-fade gap: CPU local-pose blend vs the baked component-space reference
 *     (mrw_bake_sample_xform + mrw_xform_nlerp), reported as a bounded gap.
 * (4) backend benchmark: mrw_batch_clip_to_palette over N instances on scalar / SSE2 / AVX2, with
 *     SIMD-vs-scalar parity and per-backend timing.
 *
 * All CPU-side (no GPU needed); the GPU-shader-vs-CPU readback is validate_gpu (separate). */
#ifndef DEMO_VALIDATE_H
#define DEMO_VALIDATE_H

#include "assets_proc.h"
#include "vk_context.h"
#include "marrow.h"

#include <stdio.h>

/* Run the CPU parity checks + the backend benchmark over `bench_n` instances. Prints a report to
 * stderr. Returns 0 if the exact-at-frame check passes (sub-mm) and SIMD matches scalar, else 1. */
int validate_run(const ProcAssets *assets, uint32_t bench_n);

/* Pure-CPU scalar/SSE2/AVX2 microbench of mrw_batch_clip_to_palette over N instances: prints a
 * per-backend ns/inst-bone + SIMD-vs-scalar parity table to `out`. Shared by --validate and --bench.
 * Returns 0 if SIMD output matches scalar within the visual-only tolerance. */
int validate_microbench(const mrw_skeleton_view *skel, const mrw_clip_view *clip,
                        float dur, uint32_t N, FILE *out);

/* CPU-crowd layout parity: drive the live CPU tier's batch path and assert its AoS palette row
 * layout (as skin_tierA.vert indexes it) skins identically to the per-instance mrw_clip_to_palette
 * reference - catches an AoS-row / scalar_block_layout mismatch deterministically. CPU-only, no GPU.
 * Returns 0 on pass. */
int validate_cpu_crowd_layout(const ProcAssets *assets);

/* f16 palette parity: run the batch in f32 and f16 over N instances (scalar + host-best), decode the
 * f16 output with mrw_half_to_float, and assert it matches the f32 palette within (SIMD tolerance +
 * ½ binary16 quantization) - the demo-side f16 gate. CPU-only, no GPU. Returns 0 on pass. */
int validate_cpu_crowd_f16(const ProcAssets *assets);

/* Concurrency parity: run mrw_batch_clip_to_palette serially over N instances, then run the SAME
 * work fanned across a thread pool (each lane a disjoint output slice + its own scratch, sharing the
 * read-only skeleton/clip views + dispatch), and assert the two outputs are BIT-IDENTICAL - the
 * "concurrent calls sharing read-only assets + dispatch context" gate. Covers scalar + the host-best
 * backend (SIMD output is partition-independent). CPU-only. Returns 0 on pass. */
int validate_jobs_parity(const ProcAssets *assets);

/* GPU readback check: skin the mesh on the GPU (baked compute) and compare the pre-projection
 * skinned positions against the CPU reference (mrw_baked_sample_bone LBS). Returns 0 on pass. */
int validate_gpu(VkCtx *ctx, const ProcAssets *assets);

/* f16 GPU readback: generate one instance's f16 full-matrix palette, upload it as an RGBA16F texture
 * (3 texels/joint), skin the mesh on the GPU with a direct texelFetch + LBS, and compare the skinned
 * positions against the f32 mrw_clip_to_palette reference - the gap is pure f16 quantization. The
 * end-to-end "upload RGBA16F, confirm parity" proof. Returns 0 on pass. */
int validate_gpu_f16(VkCtx *ctx, const ProcAssets *assets);

#endif /* DEMO_VALIDATE_H */
