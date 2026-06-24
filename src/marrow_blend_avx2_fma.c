/* AVX2+FMA instantiation of the pose-combine kernels (8-wide). Selected only when MRW_FEAT_FMA
 * is set; here mrw_w_fma lowers to _mm256_fmadd_ps. */
#define MRW_ISA_AVX2     1
#define MRW_ISA_FMA      1
#define MRW_BLEND_KERNEL mrw_blend_kernel_avx2_fma
#define MRW_ACCUM_KERNEL mrw_accum_kernel_avx2_fma
#include "marrow_blend_simd.h"
