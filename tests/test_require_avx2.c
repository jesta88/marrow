/* Env-gated "the SIMD job really ran AVX2" check.
 *
 * The randomized parity tests SKIP any backend the host lacks, so on a CI runner without AVX2 the AVX2
 * parity silently no-ops and the job still goes green - a "green proves nothing" trap. The CI SIMD job
 * sets MRW_REQUIRE_AVX2=1; this test then FAILS unless mrw_dispatch_detect reports the AVX2 backend with
 * FMA, proving AVX2+FMA was actually exercised. With the env var unset it is a clean, host-independent
 * pass, so a non-AVX2 host - or the in-scope Jaguar (PS4/Xbox One) 128-bit baseline - is never forced to
 * fail locally; only the job that claims to test AVX2 enforces it. */
#include "marrow.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const char *req = getenv("MRW_REQUIRE_AVX2");
    if (!req || strcmp(req, "1") != 0) {
        printf("test_require_avx2: MRW_REQUIRE_AVX2 not set - host-independent pass\n");
        return 0;
    }
    mrw_dispatch d;
    if (mrw_dispatch_detect(&d) != MRW_OK) {
        printf("test_require_avx2: FAIL - mrw_dispatch_detect returned an error\n");
        return 1;
    }
    if (d.backend != MRW_BACKEND_AVX2 || !(d.features & MRW_FEAT_FMA)) {
        printf("test_require_avx2: FAIL - required AVX2+FMA not detected (backend=%u features=0x%X)\n",
               d.backend, d.features);
        return 1;
    }
    printf("test_require_avx2: ok - AVX2+FMA exercised (backend=%u features=0x%X)\n", d.backend, d.features);
    return 0;
}
