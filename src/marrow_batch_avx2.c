/* AVX2 (no-FMA) instantiation of the batch kernel (8-wide). Selected when the host has AVX2 but
 * not FMA. Uses no _mm256_fmadd_* and is built /fp:precise (-ffp-contract=off), so it emits no
 * FMA - the distinct AVX2+FMA TU is selected only when MRW_FEAT_FMA is set. */
#define MRW_ISA_AVX2 1
#define MRW_KERNEL   mrw_batch_kernel_avx2
#include "marrow_batch_simd.h"
