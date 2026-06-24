/* The committed golden fixture: one skeleton + one clip + one baked section, with an
 * odd joint count and hardcoded inverse-bind. build_fixture() is deterministic, so the
 * in-memory build and the committed fixtures/rig.mrw must be byte-identical. */
#ifndef FIXTURE_RIG_H
#define FIXTURE_RIG_H

#include "mrw_build.h"
#include <string.h>

static size_t build_fixture(uint8_t **out) {
    static const uint16_t parent[3] = { 0xFFFF, 0, 1 };
    static const char *names[3] = { "root", "spine", "head" };
    static const float rest[3*10] = {
        0,0,0,1, 0,0,0, 1,1,1,
        0,0,0,1, 1,0,0, 1,1,1,
        0,0,0,1, 0,1,0, 1,1,1,
    };
    /* inverse of the bind model: j0=I, j1=[I|(1,0,0)], j2=[I|(1,1,0)] */
    static const float ib[3*12] = {
        1,0,0,0,  0,1,0,0,  0,0,1,0,
        1,0,0,-1, 0,1,0,0,  0,0,1,0,
        1,0,0,-1, 0,1,0,-1, 0,0,1,0,
    };
    static const float samp[3*2*10] = {
        0,0,0,1,0,0,0,1,1,1,  0,0,0,1,0,0,0,1,1,1,
        0,0,0,1,1,0,0,1,1,1,  0,0,0,1,1,0,0,1,1,1,
        0,0,0,1,0,1,0,1,1,1,  0,0,0,1,0,1,0,1,1,1,
    };
    static uint16_t tex[12*4];
    for (int i = 0; i < 12; ++i) {
        tex[i*4+0] = 0; tex[i*4+1] = 0; tex[i*4+2] = 0; tex[i*4+3] = mrw_f32_to_half(1.0f);
    }
    static const uint32_t bidx[1] = {0}, bff[1] = {0}, bfc[1] = {2}, bflags[1] = {0};
    static const float bdur[1] = { 1.0f };

    mrw_skel skel = { 3, parent, rest, ib, names };
    mrw_clip clip = { 1.0f, 2, 0, samp, NULL };
    mrw_baked baked = { 0, 2, tex, 1, bidx, bff, bfc, bdur, bflags };
    return mrw_build(&skel, &clip, 1, &baked, out);
}

#endif /* FIXTURE_RIG_H */
