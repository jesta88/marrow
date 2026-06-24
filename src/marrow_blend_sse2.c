/* SSE2 instantiation of the pose-combine kernels (4-wide; two subgroups per 8-lane tile).
 * Own translation unit so the baseline can never emit AVX. */
#define MRW_ISA_SSE2     1
#define MRW_BLEND_KERNEL mrw_blend_kernel_sse2
#define MRW_ACCUM_KERNEL mrw_accum_kernel_sse2
#include "marrow_blend_simd.h"
