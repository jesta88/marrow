/* Shared rig + clip generator for the marrow-vs-ozz comparison.
 * ONE set of float arrays (parent / rest / inverse-bind / dense samples) is generated once and
 * fed to BOTH libraries - marrow via the checked authoring lib, ozz via RawSkeleton/RawAnimation - so
 * the rig and keyframes are provably identical. Tracks are SMOOTH (sinusoidal), so ozz's key-reduction
 * axis is meaningful (random per-frame keys would not compress). Chain topology (parent[j]=j-1) means
 * marrow order == ozz depth-first order, so joint indices line up 1:1. */
#ifndef BENCH_GEN_H
#define BENCH_GEN_H

extern "C" {
#include "marrow.h"
#include "mrw_authoring.h"   /* mrw_authoring_build / _alloc / _free */
}

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "ozz/animation/offline/raw_skeleton.h"
#include "ozz/animation/offline/raw_animation.h"
#include "ozz/animation/offline/skeleton_builder.h"
#include "ozz/animation/offline/animation_builder.h"
#include "ozz/animation/offline/animation_optimizer.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/animation/runtime/animation.h"
#include "ozz/base/maths/transform.h"
#include "ozz/base/maths/quaternion.h"
#include "ozz/base/maths/vec_float.h"
#include "ozz/base/memory/unique_ptr.h"

struct RigData {
    uint32_t J = 0, sc = 0;
    float    fps = 30.0f, dur = 0.0f;
    int      loop = 1;
    std::vector<uint16_t> parent;       // J
    std::vector<float>    rest;          // J*10  (q4,t3,s3)
    std::vector<float>    ib;            // J*12  (3x4 row-major)
    std::vector<float>    samples;       // J*sc*10 joint-major (marrow layout)
    std::vector<float>    bind_pos;      // J*3   (joint origin in bind/model space - for probes)
};

/* invert a rigid-ish 3x4 affine (row-major) - ported from demo/assets_proc.c. */
static inline void bg_invert_affine12(const float m[12], float out[12]) {
    float a = m[0], b = m[1], c = m[2];
    float d = m[4], e = m[5], f = m[6];
    float g = m[8], h = m[9], i = m[10];
    float A = (e*i - f*h), B = -(d*i - f*g), C = (d*h - e*g);
    float det = a*A + b*B + c*C;
    float inv = det != 0.0f ? 1.0f / det : 0.0f;
    float mi[9];
    mi[0] = A * inv;            mi[1] = -(b*i - c*h) * inv; mi[2] =  (b*f - c*e) * inv;
    mi[3] = B * inv;            mi[4] =  (a*i - c*g) * inv; mi[5] = -(a*f - c*d) * inv;
    mi[6] = C * inv;            mi[7] = -(a*h - b*g) * inv; mi[8] =  (a*e - b*d) * inv;
    float tx = m[3], ty = m[7], tz = m[11];
    out[0] = mi[0]; out[1] = mi[1]; out[2]  = mi[2];  out[3]  = -(mi[0]*tx + mi[1]*ty + mi[2]*tz);
    out[4] = mi[3]; out[5] = mi[4]; out[6]  = mi[5];  out[7]  = -(mi[3]*tx + mi[4]*ty + mi[5]*tz);
    out[8] = mi[6]; out[9] = mi[7]; out[10] = mi[8];  out[11] = -(mi[6]*tx + mi[7]*ty + mi[8]*tz);
}

/* Build the shared arrays. Smooth per-joint rotation tracks; static translation/scale (so ozz's
 * optimizer collapses those tracks - the realistic, compressible case). */
static inline void bg_gen_rig(uint32_t J, uint32_t sc, float fps, RigData &r) {
    r.J = J; r.sc = sc; r.fps = fps; r.dur = (float)(sc - 1) / fps; r.loop = 1;
    r.parent.assign(J, 0); r.rest.assign((size_t)J * 10, 0.0f);
    r.ib.assign((size_t)J * 12, 0.0f); r.samples.assign((size_t)J * sc * 10, 0.0f);
    r.bind_pos.assign((size_t)J * 3, 0.0f);

    for (uint32_t j = 0; j < J; ++j) {
        r.parent[j] = (j == 0) ? 0xFFFFu : (uint16_t)(j - 1);
        float *rl = &r.rest[(size_t)j * 10];
        rl[0] = 0; rl[1] = 0; rl[2] = 0; rl[3] = 1;                       /* identity rest rotation */
        rl[4] = (j == 0) ? 0.0f : 0.22f;                                  /* chain offset along +X  */
        rl[5] = (j == 0) ? 1.0f : 0.06f * (float)((j % 3) - 1);
        rl[6] = 0.0f;
        rl[7] = rl[8] = rl[9] = 1.0f;                                     /* unit scale */
    }

    /* inverse-bind = inverse of composed bind model (reuse marrow's public math for the compose). */
    std::vector<float> model((size_t)J * 12, 0.0f);
    for (uint32_t j = 0; j < J; ++j) {
        mrw_trs trs;
        std::memcpy(trs.rot,   &r.rest[(size_t)j*10 + 0], 4 * sizeof(float));
        std::memcpy(trs.trans, &r.rest[(size_t)j*10 + 4], 3 * sizeof(float));
        std::memcpy(trs.scale, &r.rest[(size_t)j*10 + 7], 3 * sizeof(float));
        float local[12];
        mrw_trs_to_affine(&trs, local);
        float *m = &model[(size_t)j * 12];
        if (r.parent[j] == 0xFFFFu) std::memcpy(m, local, sizeof(float) * 12);
        else                        mrw_affine_mul(&model[(size_t)r.parent[j] * 12], local, m);
        bg_invert_affine12(m, &r.ib[(size_t)j * 12]);
        r.bind_pos[(size_t)j*3+0] = m[3]; r.bind_pos[(size_t)j*3+1] = m[7]; r.bind_pos[(size_t)j*3+2] = m[11];
    }

    /* smooth sinusoidal rotation per joint; constant translation (= rest) and scale. */
    for (uint32_t j = 0; j < J; ++j) {
        float amp   = 0.30f + 0.20f * sinf((float)j * 0.7f);              /* radians */
        float phase = (float)j * 0.9f;
        int   axis  = (int)(j % 3);                                       /* x/y/z rotation axis */
        for (uint32_t s = 0; s < sc; ++s) {
            float ph = (float)s / (float)(sc - 1);                        /* [0,1] */
            float ang = amp * sinf(6.2831853f * ph + phase);
            float ha = ang * 0.5f, sn = sinf(ha), cs = cosf(ha);
            float q[4] = { 0, 0, 0, cs }; q[axis] = sn;
            float *o = &r.samples[((size_t)j * sc + s) * 10];
            o[0] = q[0]; o[1] = q[1]; o[2] = q[2]; o[3] = q[3];
            o[4] = r.rest[(size_t)j*10 + 4]; o[5] = r.rest[(size_t)j*10 + 5]; o[6] = r.rest[(size_t)j*10 + 6];
            o[7] = 1.0f; o[8] = 1.0f; o[9] = 1.0f;
        }
    }
}

/* marrow: author a {skeleton, 1 clip} blob from the shared arrays. Caller frees with
 * mrw_authoring_free. Returns nullptr on failure. */
static inline uint8_t *bg_build_marrow(const RigData &r, size_t *out_size) {
    static thread_local std::vector<std::string> name_store;
    name_store.clear();
    name_store.reserve(r.J);
    std::vector<const char *> names(r.J);
    for (uint32_t j = 0; j < r.J; ++j) { name_store.push_back("j" + std::to_string(j)); }
    for (uint32_t j = 0; j < r.J; ++j) names[j] = name_store[j].c_str();

    mrw_skel skel; skel.joint_count = r.J; skel.parent = r.parent.data();
    skel.rest_local = r.rest.data(); skel.inverse_bind = r.ib.data(); skel.names = names.data();

    mrw_clip clip; clip.fps = r.fps; clip.sample_count = r.sc;
    clip.flags = r.loop ? MRW_CLIP_LOOPING : 0u;
    clip.samples = r.samples.data(); clip.root_track = nullptr;

    uint8_t *buf = nullptr; size_t sz = 0;
    if (mrw_authoring_build(&skel, &clip, 1, nullptr, &buf, &sz) != MRW_OK) return nullptr;
    *out_size = sz; return buf;
}

/* ozz: build the runtime skeleton from the shared arrays (chain hierarchy). */
static inline ozz::unique_ptr<ozz::animation::Skeleton> bg_build_ozz_skeleton(const RigData &r) {
    using namespace ozz::animation::offline;
    using ozz::math::Float3; using ozz::math::Quaternion;
    RawSkeleton raw;
    raw.roots.resize(1);
    RawSkeleton::Joint *cur = &raw.roots[0];
    for (uint32_t j = 0; j < r.J; ++j) {
        cur->name = ("j" + std::to_string(j)).c_str();
        const float *rl = &r.rest[(size_t)j * 10];
        cur->transform.rotation    = Quaternion(rl[0], rl[1], rl[2], rl[3]);
        cur->transform.translation = Float3(rl[4], rl[5], rl[6]);
        cur->transform.scale       = Float3(rl[7], rl[8], rl[9]);
        if (j + 1 < r.J) { cur->children.resize(1); cur = &cur->children[0]; }
    }
    SkeletonBuilder builder;
    return builder(raw);
}

/* Fill a RawAnimation (a key per frame for every component) from the shared arrays. */
static inline void bg_build_raw_animation(const RigData &r, ozz::animation::offline::RawAnimation &raw) {
    using namespace ozz::animation::offline;
    using ozz::math::Float3; using ozz::math::Quaternion;
    raw.duration = r.dur;
    raw.tracks.resize(r.J);
    for (uint32_t j = 0; j < r.J; ++j) {
        RawAnimation::JointTrack &tr = raw.tracks[j];
        tr.translations.resize(r.sc); tr.rotations.resize(r.sc); tr.scales.resize(r.sc);
        for (uint32_t s = 0; s < r.sc; ++s) {
            float t = (float)s / r.fps;
            const float *o = &r.samples[((size_t)j * r.sc + s) * 10];
            tr.translations[s] = { t, Float3(o[4], o[5], o[6]) };
            tr.rotations[s]    = { t, Quaternion(o[0], o[1], o[2], o[3]) };
            tr.scales[s]       = { t, Float3(o[7], o[8], o[9]) };
        }
    }
}

static inline size_t bg_count_keys(const ozz::animation::offline::RawAnimation &raw) {
    size_t k = 0;
    for (size_t j = 0; j < raw.tracks.size(); ++j)
        k += raw.tracks[j].translations.size() + raw.tracks[j].rotations.size() + raw.tracks[j].scales.size();
    return k;
}

/* ozz: build the runtime animation from the shared arrays. optimize=true runs the key-reduction
 * AnimationOptimizer pass first (ozz-opt); false keeps a key per frame (ozz-dense). When opt_keys is
 * non-null it receives the post-optimization key count (for the memory table). */
static inline ozz::unique_ptr<ozz::animation::Animation> bg_build_ozz_animation(
        const RigData &r, bool optimize, const ozz::animation::Skeleton &skel, size_t *out_keys = nullptr) {
    using namespace ozz::animation::offline;
    RawAnimation raw;
    bg_build_raw_animation(r, raw);
    AnimationBuilder builder;
    if (optimize) {
        AnimationOptimizer optimizer;
        RawAnimation opt;
        if (!optimizer(raw, skel, &opt)) return ozz::unique_ptr<ozz::animation::Animation>();
        if (out_keys) *out_keys = bg_count_keys(opt);
        return builder(opt);
    }
    if (out_keys) *out_keys = bg_count_keys(raw);
    return builder(raw);
}

#endif /* BENCH_GEN_H */
