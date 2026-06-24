/* SSE2 instantiation of the batch kernel (4-wide; two subgroups per 8-lane tile). Compiled in
 * its own translation unit so the baseline can never emit AVX. */
#define MRW_ISA_SSE2 1
#define MRW_KERNEL   mrw_batch_kernel_sse2
#include "marrow_batch_simd.h"
