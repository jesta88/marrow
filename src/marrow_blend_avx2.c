/* AVX2 (no-FMA) instantiation of the pose-combine kernels (8-wide). Selected when the host has
 * AVX2 but not FMA; built /fp:precise (-ffp-contract=off) so it emits no FMA. */
#define MRW_ISA_AVX2     1
#define MRW_BLEND_KERNEL mrw_blend_kernel_avx2
#define MRW_ACCUM_KERNEL mrw_accum_kernel_avx2
#include "marrow_blend_simd.h"
