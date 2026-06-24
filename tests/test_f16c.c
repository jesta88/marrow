/* Encoder gate: the portable RNE encoder
 * mrw_f32_to_f16 (src/marrow_math.c) must be BIT-IDENTICAL to the F16C vcvtps2ph instruction over a
 * wide sample of finite f32 - so the AVX2 f16 palette store (vcvtps2ph) and the scalar/SSE2 fallback
 * (mrw_f32_to_f16) produce the same bytes. This TU is compiled with F16C; the sweep runs only when the
 * host actually has F16C (detected via the public dispatch), else it skips cleanly. */
#include "test_util.h"
#include <immintrin.h>

#define RND _MM_FROUND_TO_NEAREST_INT   /* RNE; matches the kernel's vcvtps2ph rounding immediate */

static void check_one(float x) {
    uint16_t a = mrw_f32_to_f16(x);
    uint16_t b = (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), RND), 0); /* hardware vcvtps2ph, RNE */
    if (a != b) {
        uint32_t bits; memcpy(&bits, &x, 4);
        printf("FAIL f16c-bitident: x=%g (0x%08X): mrw=0x%04X vcvtps2ph=0x%04X\n", x, bits, a, b);
        ++g_fail;
    }
}

int main(void) {
    mrw_dispatch d; mrw_dispatch_detect(&d);
    if (!(d.features & MRW_FEAT_F16C)) { printf("test_f16c: skipped (host has no F16C)\n"); return 0; }

    /* Structured sweep over the finite f32 space: every biased exponent (0 = subnormal/zero through
     * 254 = max finite), a prime-strided set of mantissas per exponent (covers RNE ties, the f16
     * normal/subnormal/overflow boundaries, and the round-up-into-exponent carry), both signs. */
    for (uint32_t exp = 0; exp <= 254u && !g_fail; ++exp) {
        for (uint32_t m = 0; m < 0x800000u; m += 0x1001u) {
            uint32_t bits = (exp << 23) | m;
            float x; memcpy(&x, &bits, 4);
            check_one(x);
            uint32_t nbits = bits | 0x80000000u;
            float nx; memcpy(&nx, &nbits, 4);
            check_one(nx);
        }
    }
    /* Exact edges that the strided loop can miss: ±0, mantissa extremes per exponent, and a few
     * literals straddling f16 rounding boundaries. */
    check_one(0.0f); check_one(-0.0f);
    check_one(1.0f + (float)ldexp(1.0, -11)); /* tie → even (down) */
    check_one(1.0f + (float)ldexp(3.0, -11)); /* tie → even (up)   */
    check_one(65504.0f);                       /* largest finite half */
    check_one(65520.0f);                       /* rounds to inf       */
    check_one((float)ldexp(1.0, -25));         /* rounds to 0 (subnormal underflow) */

    printf(g_fail ? "test_f16c: %d FAILED\n" : "test_f16c: ok\n", g_fail);
    TEST_MAIN_RETURN();
}
