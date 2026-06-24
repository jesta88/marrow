/* Synthetic "compute-only" floor.
 *
 * Compiles a SECOND copy of the across-instance AVX2+FMA kernel from the single shared source
 * (src/marrow_batch_simd.h) with MRW_BENCH_NO_SCATTER defined: identical gather + nlerp + compose +
 * inverse-bind fold, but the 204 MB SoA→AoS palette write is replaced by a thread-local accumulate.
 * Timing this against the real kernel brackets how much of the heavy-rig pooled time is the output
 * write itself vs pure compute. Bench-only: this TU lives in the gated ozz-bench target and is NEVER
 * linked into the marrow runtime, which keeps src/ byte-identical (MRW_BENCH_NO_SCATTER is undefined
 * in every runtime/test/demo build). */
#define MRW_ISA_AVX2
#define MRW_ISA_FMA
#define MRW_BENCH_NO_SCATTER
#define MRW_KERNEL mrw_floor_noscatter_avx2_fma
#include "marrow_batch_simd.h"
