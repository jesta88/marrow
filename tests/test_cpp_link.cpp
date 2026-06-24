/* Verifies the public header compiles as C++ and links against the C library
 * (extern "C" ABI) - the library compiles and links as both C11 and C++. */
#include "marrow.h"
#include <cstdio>

int main() {
    mrw_dispatch d;
    if (mrw_dispatch_scalar(&d) != MRW_OK) return 1;
    if (d.backend != MRW_BACKEND_SCALAR) return 1;

    mrw_dispatch det;
    if (mrw_dispatch_detect(&det) != MRW_OK) return 1;

    if (mrw_half_to_float(0x3C00) != 1.0f) return 1;

    float q[4] = { 0, 0, 0, 1 }, r[9];
    mrw_quat_to_mat3(q, r);
    if (r[0] != 1.0f || r[4] != 1.0f || r[8] != 1.0f) return 1;

    /* batch surface compiles and links as C++ */
    mrw_mem_req sreq, oreq;
    if (mrw_batch_clip_to_palette_requirements(8, 100, MRW_PALETTE_F32, &sreq, &oreq) != MRW_OK) return 1;
    if (sreq.align != 64 || oreq.align != 16) return 1;
    if (mrw_batch_clip_to_palette(&d, nullptr, nullptr, nullptr, 0, nullptr, 0, nullptr, 0)
        != MRW_E_RANGE) return 1; /* disp ok, skel NULL ⇒ RANGE - exercises the symbol */

    /* pose algebra: real calls on a 1-joint identity pose */
    mrw_trs pa = { {0,0,0,1}, {0,0,0}, {1,1,1} };
    mrw_trs pb = { {0,0,0,1}, {1,0,0}, {2,2,2} };
    mrw_trs pout, pdelta;
    if (mrw_pose_blend(&pa, &pb, 0.5f, nullptr, &pout, 1, 1) != MRW_OK) return 1;
    if (mrw_pose_make_additive(&pb, &pa, &pdelta, 1, 1) != MRW_OK) return 1;
    if (mrw_pose_accumulate(&pa, &pdelta, 1.0f, nullptr, &pout, 1, 1) != MRW_OK) return 1;

    /* IK: null-skel RANGE return - exercises (links + calls) both symbols */
    float tgt[3] = { 1, 0, 0 }, pole[3] = { 0, 1, 0 };
    if (mrw_ik_two_bone(nullptr, nullptr, nullptr, 0, 1, 2, tgt, pole, 1.0f, 3) != MRW_E_RANGE) return 1;
    float aimax[3] = { 0, 0, 1 }, upax[3] = { 0, 1, 0 }, upm[3] = { 0, 1, 0 };
    if (mrw_ik_aim(nullptr, nullptr, nullptr, 0, aimax, upax, tgt, upm, 1.0f, 1) != MRW_E_RANGE) return 1;

    /* batch pose ops: requirements queries link + return sizes */
    mrw_mem_req bsreq, boreq;
    if (mrw_batch_blend_clips_to_palette_requirements(8, 100, MRW_PALETTE_F32, &bsreq, &boreq) != MRW_OK) return 1;
    if (mrw_batch_accumulate_to_palette_requirements(8, 100, MRW_PALETTE_F16, &bsreq, &boreq) != MRW_OK) return 1;
    if (mrw_batch_blend_clips_to_palette(&d, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                         0, nullptr, 0, nullptr, 0) != MRW_E_RANGE) return 1;
    if (mrw_batch_accumulate_to_palette(&d, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        0, nullptr, 0, nullptr, 0) != MRW_E_RANGE) return 1;

    std::printf("test_cpp_link: ok (scalar backend, host features=0x%x)\n", det.features);
    return 0;
}
