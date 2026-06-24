/* marrow vs ozz-animation - CPU throughput comparison harness.
 *
 * Methodology: we produce, for N instances of ONE shared rig at per-instance
 * times, the per-instance per-joint skinning palette (model x inverse_bind) - marrow in one fused
 * across-instance batch call, ozz as a per-instance SamplingJob -> LocalToModelJob -> inverse-bind
 * pipeline - and compare throughput. Both write the SAME canonical 3x4 palette. A correctness gate
 * (shared probe skinning, sub-mm) runs BEFORE any timing; a no-allocation gate fails if anything
 * allocates inside a measured window. Timing uses demo/profiler.c's prof_now_s + the public prof_stat
 * percentile machinery; the demo thread pool drives Regime B (multithread). */

#define _CRT_SECURE_NO_WARNINGS   /* MSVC: the glTF path uses fopen() to read the .mrw blob */

#include "bench_gen.h"

extern "C" {
#include "profiler.h"   /* prof_now_s, prof_stat, prof_mean, prof_pct */
#include "jobs.h"       /* thread pool (Regime B) */
}

#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/base/memory/allocator.h"
#include "ozz/base/span.h"

#ifdef MRW_OZZ_BENCH_GLTF
/* Real-asset (UAL1) ecological cross-check. Each library goes through
 * its OWN offline importer: marrow's gltf2marrow (→ .mrw, loaded here) and ozz's gltf2ozz (→ .ozz,
 * loaded here via IArchive). The skin inverse-bind matrices - which ozz's glTF path discards - are
 * read from the glTF with cgltf and applied to ozz with the SAME column-major→3×4 conversion
 * gltf2marrow uses, so both multiply by identical inverse-bind. The timed runtime path is unchanged. */
#include "ozz/base/io/archive.h"
#include "ozz/base/io/stream.h"
#include <string>
extern "C" {
#include "joint_order.h"   /* mrw_g2m_joint_order_build - the exact skin→marrow DFS gltf2marrow/the demo use */
}
#endif

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <immintrin.h>   /* pure-streaming floor stores (_mm256_storeu_ps) */
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

typedef ozz::animation::SamplingJob::Context OzzCtx;

/* ------------------------------------------------------------------ allocation accounting */

static std::atomic<long long> g_alloc_count{0};
static std::atomic<long long> g_alloc_bytes{0};

void *operator new(std::size_t n) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add((long long)n, std::memory_order_relaxed);
    void *p = std::malloc(n ? n : 1); if (!p) throw std::bad_alloc(); return p;
}
void *operator new[](std::size_t n) { return operator new(n); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

/* ozz routes its allocations through this; wraps the real allocator so we both count and forward. */
class CountingAllocator : public ozz::memory::Allocator {
  public:
    explicit CountingAllocator(ozz::memory::Allocator *base) : base_(base) {}
    void *Allocate(size_t size, size_t align) override {
        g_alloc_count.fetch_add(1, std::memory_order_relaxed);
        g_alloc_bytes.fetch_add((long long)size, std::memory_order_relaxed);
        return base_->Allocate(size, align);
    }
    void Deallocate(void *block) override { base_->Deallocate(block); }
  private:
    ozz::memory::Allocator *base_;
};
static long long alloc_count() { return g_alloc_count.load(std::memory_order_relaxed); }
static long long alloc_bytes() { return g_alloc_bytes.load(std::memory_order_relaxed); }

/* ------------------------------------------------------------------ small math helpers */

/* out = A o B for 3x4 row-major affines (A after B) - the inverse-bind multiply an ozz integrator
 * writes when targeting a 3x4 palette (ozz-affine3x4). Same FLOPs as marrow's fused multiply. */
static inline void mul_affine3x4(const float A[12], const float B[12], float out[12]) {
    for (int r = 0; r < 3; ++r) {
        const float a0 = A[r*4+0], a1 = A[r*4+1], a2 = A[r*4+2], a3 = A[r*4+3];
        out[r*4+0] = a0*B[0] + a1*B[4] + a2*B[8];
        out[r*4+1] = a0*B[1] + a1*B[5] + a2*B[9];
        out[r*4+2] = a0*B[2] + a1*B[6] + a2*B[10];
        out[r*4+3] = a0*B[3] + a1*B[7] + a2*B[11] + a3;
    }
}

/* Float4x4 (column-major) -> 3x4 row-major affine (12 floats), dropping the [0,0,0,1] row. */
static inline void extract3x4(const ozz::math::Float4x4 &M, float out[12]) {
    float c0[4], c1[4], c2[4], c3[4];
    ozz::math::StorePtrU(M.cols[0], c0);
    ozz::math::StorePtrU(M.cols[1], c1);
    ozz::math::StorePtrU(M.cols[2], c2);
    ozz::math::StorePtrU(M.cols[3], c3);
    out[0] = c0[0]; out[1] = c1[0]; out[2]  = c2[0]; out[3]  = c3[0];
    out[4] = c0[1]; out[5] = c1[1]; out[6]  = c2[1]; out[7]  = c3[1];
    out[8] = c0[2]; out[9] = c1[2]; out[10] = c2[2]; out[11] = c3[2];
}

static inline ozz::math::Float4x4 ib_to_float4x4(const float ib[12]) {
    using ozz::math::simd_float4::Load;
    ozz::math::Float4x4 F;
    F.cols[0] = Load(ib[0], ib[4], ib[8],  0.0f);
    F.cols[1] = Load(ib[1], ib[5], ib[9],  0.0f);
    F.cols[2] = Load(ib[2], ib[6], ib[10], 0.0f);
    F.cols[3] = Load(ib[3], ib[7], ib[11], 1.0f);
    return F;
}

static inline void advance_times(float *t, uint32_t N, float dt, float dur) {
    for (uint32_t i = 0; i < N; ++i) { float x = t[i] + dt; if (x >= dur) x -= dur; t[i] = x; }
}

/* ------------------------------------------------------------------ ozz palette pipeline */

enum OzzMode { OZZ_MODEL, OZZ_AFFINE3X4, OZZ_MUL4X4 };

struct OzzScratch {
    std::vector<ozz::math::SoaTransform> locals;
    std::vector<ozz::math::Float4x4>     models;
    void init(int num_soa, int J) { locals.resize(num_soa); models.resize(J); }
};

struct OzzRig {
    const ozz::animation::Skeleton  *skel = nullptr;
    const ozz::animation::Animation *anim = nullptr;
    int J = 0, num_soa = 0;
    int model_count = 0;                    // ozz model joints (== J for the procedural chain;
                                            // > J for the glTF rig, which delivers J skin joints
                                            // out of a deeper hierarchy - see model_idx)
    const uint32_t *model_idx = nullptr;    // output slot j -> ozz model-joint index, or null = identity
    float dur = 0.0f;
    std::vector<float> ib;                  // J*12  (inverse-bind in OUTPUT-slot order)
    std::vector<ozz::math::Float4x4> ib4;   // J
};

/* Produce palettes for instances [begin,end) at times[]; sc is this lane's scratch; ctxp[i] is
 * instance i's warm Context. Writes out[(i*J+j)*12]. */
static void ozz_palette_range(const OzzRig &R, OzzCtx **ctxp, const float *times,
                              uint32_t begin, uint32_t end, OzzScratch &sc, float *out, OzzMode mode) {
    const int J  = R.J;                                  /* output skin joints (the deliverable) */
    const int MC = R.model_count;                        /* ozz model joints sampled/local-to-modeled */
    const uint32_t *mi = R.model_idx;                    /* output slot -> model index, or null = identity */
    ozz::math::SoaTransform *locals = sc.locals.data();
    ozz::math::Float4x4     *models = sc.models.data();
    for (uint32_t i = begin; i < end; ++i) {
        float ratio = times[i] / R.dur;                 /* seconds -> ratio (in-kernel) */
        if (ratio < 0.0f) ratio = 0.0f; else if (ratio >= 1.0f) ratio -= (float)(int)ratio;

        ozz::animation::SamplingJob sj;
        sj.animation = R.anim; sj.context = ctxp[i]; sj.ratio = ratio;
        sj.output = ozz::span<ozz::math::SoaTransform>(locals, (size_t)R.num_soa);
        sj.Run();

        ozz::animation::LocalToModelJob l2m;
        l2m.skeleton = R.skel;
        l2m.input  = ozz::span<const ozz::math::SoaTransform>(locals, (size_t)R.num_soa);
        l2m.output = ozz::span<ozz::math::Float4x4>(models, (size_t)MC);
        l2m.Run();

        float *po = &out[(size_t)i * J * 12];
        /* Branch on the remap ONCE per instance, not per joint: the !mi arm is byte-identical to the
         * original frozen loop, so the procedural baseline's timed path is unchanged (the glTF rig is
         * the only caller that sets model_idx). */
        if (!mi) {
            for (int j = 0; j < J; ++j) {
                if (mode == OZZ_MUL4X4) {
                    ozz::math::Float4x4 pal = models[j] * R.ib4[j];
                    extract3x4(pal, &po[j * 12]);
                } else {
                    float m3[12]; extract3x4(models[j], m3);
                    if (mode == OZZ_MODEL) std::memcpy(&po[j * 12], m3, 12 * sizeof(float));
                    else                   mul_affine3x4(m3, &R.ib[(size_t)j * 12], &po[j * 12]);
                }
            }
        } else {
            for (int j = 0; j < J; ++j) {
                const int src = (int)mi[j];              /* skin slot j -> ozz model joint */
                if (mode == OZZ_MUL4X4) {
                    ozz::math::Float4x4 pal = models[src] * R.ib4[j];
                    extract3x4(pal, &po[j * 12]);
                } else {
                    float m3[12]; extract3x4(models[src], m3);
                    if (mode == OZZ_MODEL) std::memcpy(&po[j * 12], m3, 12 * sizeof(float));
                    else                   mul_affine3x4(m3, &R.ib[(size_t)j * 12], &po[j * 12]);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ adaptive-window timing */

struct Stat { double mean_ms = 0, p95_ms = 0; };

static void stat_fill(prof_stat *s, double ms) {
    s->samples[s->head] = ms;
    s->head = (s->head + 1u) % PROF_RING;
    if (s->count < PROF_RING) s->count++;
}

/* Each window repeats `fn` enough to cover a target work amount; records per-call average as one
 * sample; collect `windows` after warm-up. fn() performs ONE frame of work over all N instances. */
template <typename Fn>
static Stat measure(uint32_t N, Fn &&fn, bool smoke) {
    /* instances per window: enough to dwarf timer noise (~ms windows) without each window being huge.
     * reps scales down with N so wall time per measure is ~constant across the sweep. */
    const uint64_t target = smoke ? 200000ull : 300000ull;
    uint32_t reps = (uint32_t)(target / (N ? N : 1)); if (reps < 1) reps = 1;
    const int windows = smoke ? 8 : 48;
    const int warm = smoke ? 2 : 6;
    for (int w = 0; w < warm; ++w) for (uint32_t r = 0; r < reps; ++r) fn();
    prof_stat st; std::memset(&st, 0, sizeof st);
    for (int w = 0; w < windows; ++w) {
        double t0 = prof_now_s();
        for (uint32_t r = 0; r < reps; ++r) fn();
        stat_fill(&st, (prof_now_s() - t0) / (double)reps * 1000.0);
    }
    return Stat{ prof_mean(&st), prof_pct(&st, 0.95) };
}
static double ns_ij(double mean_ms, uint32_t N, uint32_t J) { return mean_ms * 1e6 / ((double)N * J); }

/* ------------------------------------------------------------------ aligned scratch */

static void *al64(size_t n) { return mrw_authoring_alloc(n); }
static void  fr(void *p)    { mrw_authoring_free(p); }
static bool  aligned(const void *p, size_t a) { return ((uintptr_t)p & (a - 1)) == 0; }

/* ------------------------------------------------------------------ marrow side */

struct MarrowRig {
    uint8_t *blob = nullptr; size_t blob_size = 0;
    mrw_blob b; mrw_skeleton_view skel; mrw_clip_view clip; uint32_t J = 0;
};
static bool marrow_build(const RigData &r, MarrowRig &m) {
    m.blob = bg_build_marrow(r, &m.blob_size);
    if (!m.blob) { std::fprintf(stderr, "[marrow] authoring failed\n"); return false; }
    if (mrw_blob_open(m.blob, m.blob_size, &m.b) != MRW_OK) { std::fprintf(stderr, "[marrow] open failed\n"); return false; }
    mrw_blob_skeleton(&m.b, &m.skel);
    if (mrw_clip_view_at(&m.b, 1, &m.clip) != MRW_OK) { std::fprintf(stderr, "[marrow] clip view failed\n"); return false; }
    m.J = r.J; return true;
}

/* ------------------------------------------------------------------ correctness gate */

static double palette_probe_diff(const RigData &r, const float *palA, const float *palB, uint32_t N) {
    double maxmm = 0.0;
    static const float off[3][3] = { {0,0,0}, {0.05f,0,0}, {0,0.05f,0.03f} };
    for (uint32_t i = 0; i < N; ++i)
        for (uint32_t j = 0; j < r.J; ++j) {
            const float *Ma = &palA[(size_t)(i*r.J + j) * 12];
            const float *Mb = &palB[(size_t)(i*r.J + j) * 12];
            for (int k = 0; k < 3; ++k) {
                float p[3] = { r.bind_pos[j*3]+off[k][0], r.bind_pos[j*3+1]+off[k][1], r.bind_pos[j*3+2]+off[k][2] };
                float a[3], b[3];
                for (int rr = 0; rr < 3; ++rr) {
                    a[rr] = Ma[rr*4]*p[0] + Ma[rr*4+1]*p[1] + Ma[rr*4+2]*p[2] + Ma[rr*4+3];
                    b[rr] = Mb[rr*4]*p[0] + Mb[rr*4+1]*p[1] + Mb[rr*4+2]*p[2] + Mb[rr*4+3];
                }
                double dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
                double mm = std::sqrt(dx*dx+dy*dy+dz*dz)*1000.0;
                if (mm > maxmm) maxmm = mm;
            }
        }
    return maxmm;
}

/* ------------------------------------------------------------------ assets bundle */

struct Assets {
    RigData rig;
    MarrowRig mr;
    ozz::unique_ptr<ozz::animation::Skeleton>  skel;
    ozz::unique_ptr<ozz::animation::Animation> anim_opt, anim_dense;
    OzzRig orig;
    size_t keys_opt = 0, keys_dense = 0;
};

static bool build_assets(uint32_t J, uint32_t sc, float fps, Assets &a) {
    bg_gen_rig(J, sc, fps, a.rig);
    if (!marrow_build(a.rig, a.mr)) return false;
    a.skel = bg_build_ozz_skeleton(a.rig);
    if (!a.skel) { std::fprintf(stderr, "[ozz] skeleton build failed\n"); return false; }
    a.anim_opt   = bg_build_ozz_animation(a.rig, true,  *a.skel, &a.keys_opt);
    a.anim_dense = bg_build_ozz_animation(a.rig, false, *a.skel, &a.keys_dense);
    if (!a.anim_opt || !a.anim_dense) { std::fprintf(stderr, "[ozz] animation build failed\n"); return false; }
    a.orig.skel = a.skel.get(); a.orig.anim = a.anim_opt.get();
    a.orig.J = (int)J; a.orig.model_count = (int)J;     /* chain: output joints == model joints */
    a.orig.num_soa = a.skel->num_soa_joints(); a.orig.dur = a.rig.dur;
    a.orig.ib.assign(a.rig.ib.begin(), a.rig.ib.end());
    a.orig.ib4.resize(J);
    for (uint32_t j = 0; j < J; ++j) a.orig.ib4[j] = ib_to_float4x4(&a.rig.ib[(size_t)j*12]);
    return true;
}

/* Build N warm per-instance ozz contexts; report measured bytes/context. */
struct Contexts {
    std::vector<std::unique_ptr<OzzCtx>> owned;
    std::vector<OzzCtx*> ptr;
    long long bytes_per_ctx = 0;
};
static void build_contexts(uint32_t N, int J, Contexts &c) {
    long long b0 = alloc_bytes();
    c.owned.resize(N); c.ptr.resize(N);
    for (uint32_t i = 0; i < N; ++i) { c.owned[i] = std::make_unique<OzzCtx>(J); c.ptr[i] = c.owned[i].get(); }
    c.bytes_per_ctx = N ? (alloc_bytes() - b0) / (long long)N : 0;
}

/* ------------------------------------------------------------------ Regime B (threading) */

struct MarrowJobCtx {
    const mrw_dispatch *disp; const mrw_skeleton_view *skel; const mrw_clip_view *clip;
    const float *times; float *out; size_t out_bytes; void *scratch; size_t unit; uint32_t J;
};
static void marrow_job(void *v, uint32_t worker, uint32_t begin, uint32_t end) {
    MarrowJobCtx *c = (MarrowJobCtx *)v; if (end <= begin) return;
    size_t off = (size_t)begin * c->J * 12u;
    void *scr = (char *)c->scratch + (size_t)worker * c->unit;
    mrw_batch_clip_to_palette(c->disp, c->skel, c->clip, &c->times[begin], end - begin,
                              c->out + off, c->out_bytes - off * sizeof(float), scr, c->unit);
}
struct OzzJobCtx { const OzzRig *R; OzzCtx **ctxp; const float *times; float *out; OzzScratch *lanes; OzzMode mode; };
static void ozz_job(void *v, uint32_t worker, uint32_t begin, uint32_t end) {
    OzzJobCtx *c = (OzzJobCtx *)v;
    ozz_palette_range(*c->R, c->ctxp, c->times, begin, end, c->lanes[worker], c->out, c->mode);
}

/* ============================================================ real-asset (UAL1) glTF path */
#ifdef MRW_OZZ_BENCH_GLTF

/* glTF column-major 4×4 → marrow 3×4 row-major (drop the implicit [0,0,0,1] row). VERBATIM the
 * conversion gltf2marrow applies to skin inverse-bind matrices (tools/gltf2marrow/convert.c), so both
 * libraries end up multiplying by the SAME inverse-bind. */
static void g_colmajor4x4_to_affine(const float m[16], float out12[12]) {
    for (int r = 0; r < 3; ++r) {
        out12[r*4+0] = m[0+r];  out12[r*4+1] = m[4+r];
        out12[r*4+2] = m[8+r];  out12[r*4+3] = m[12+r];
    }
}

/* Read a whole file into a fresh 64-aligned buffer (mrw_blob_open requires ≥64-byte alignment). */
static uint8_t *read_file_64(const char *path, size_t *out_size) {
    FILE *f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "[gltf] cannot open %s\n", path); return nullptr; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    if (n <= 0) { std::fclose(f); std::fprintf(stderr, "[gltf] empty/unreadable %s\n", path); return nullptr; }
    uint8_t *buf = (uint8_t *)mrw_authoring_alloc((size_t)n);
    if (!buf) { std::fclose(f); std::fprintf(stderr, "[gltf] OOM reading %s\n", path); return nullptr; }
    size_t got = std::fread(buf, 1, (size_t)n, f);
    std::fclose(f);
    if (got != (size_t)n) { mrw_authoring_free(buf); std::fprintf(stderr, "[gltf] short read %s\n", path); return nullptr; }
    *out_size = (size_t)n; return buf;
}

static bool ozz_load_skeleton(const char *path, ozz::animation::Skeleton *s) {
    ozz::io::File f(path, "rb");
    if (!f.opened()) { std::fprintf(stderr, "[ozz] cannot open %s\n", path); return false; }
    ozz::io::IArchive a(&f);
    if (!a.TestTag<ozz::animation::Skeleton>()) { std::fprintf(stderr, "[ozz] %s is not a Skeleton archive\n", path); return false; }
    a >> *s; return true;
}
static bool ozz_load_animation(const char *path, ozz::animation::Animation *an) {
    ozz::io::File f(path, "rb");
    if (!f.opened()) { std::fprintf(stderr, "[ozz] cannot open %s\n", path); return false; }
    ozz::io::IArchive a(&f);
    if (!a.TestTag<ozz::animation::Animation>()) { std::fprintf(stderr, "[ozz] %s is not an Animation archive\n", path); return false; }
    a >> *an; return true;
}

/* DIRECT 3×4 element-wise palette comparison (like demo/validate.c palette_diff): max abs
 * diff over the 9 basis terms, and max translation distance in mm, across all instances × joints. */
static void palette_matrix_diff(const float *A, const float *B, uint32_t N, uint32_t J,
                                double *out_max_basis, double *out_max_trans_mm) {
    static const int basis_idx[9] = { 0,1,2, 4,5,6, 8,9,10 };
    double mb = 0.0, mt = 0.0;
    for (size_t e = 0; e < (size_t)N * J; ++e) {
        const float *a = &A[e*12], *b = &B[e*12];
        for (int k = 0; k < 9; ++k) { double d = std::fabs((double)a[basis_idx[k]] - (double)b[basis_idx[k]]); if (d > mb) mb = d; }
        double dx = (double)a[3]-b[3], dy = (double)a[7]-b[7], dz = (double)a[11]-b[11];
        double tm = std::sqrt(dx*dx + dy*dy + dz*dz) * 1000.0;
        if (tm > mt) mt = tm;
    }
    *out_max_basis = mb; *out_max_trans_mm = mt;
}

/* ====================== synthetic floors ======================
 * Two measurements that bracket how much of the heavy-rig pooled time is pure compute vs the 204 MB
 * output write. (1) compute-only: the REAL AVX2+FMA kernel built from the shared src with the palette
 * write removed (floor_noscatter.c, MRW_BENCH_NO_SCATTER). (2) pure-streaming: the palette write in the
 * exact per-instance AoS stride, with no math. Together they turn the streaming-write payoff into a number. */
/* Signature MUST track MRW_KERNEL in src/marrow_batch_simd.h - the out_format/use_f16c params arrived
 * with the f16 output; the no-scatter path ignores them but the C ABI still requires them. */
extern "C" mrw_result mrw_floor_noscatter_avx2_fma(
    const mrw_skeleton_view *skel, const mrw_clip_view *clip,
    const float *times, uint32_t instance_count, void *out_palettes,
    mrw_palette_format out_format, int use_f16c, float *model);

/* pure-streaming floor: write the full palette over [begin,end) in the kernel's (tile-of-8, joint,
 * lane) store order at the real per-instance stride - the DRAM write ceiling, no pose math. */
static void floor_stream(float *out, uint32_t begin, uint32_t end, uint32_t J) {
    const __m256 v8 = _mm256_set1_ps(1.0f);
    const __m128 v4 = _mm_set1_ps(1.0f);
    const size_t s = (size_t)J * 12;                 /* per-instance stride (floats) */
    for (uint32_t base = begin; base < end; base += 8) {
        uint32_t live = end - base; if (live > 8) live = 8;
        for (uint32_t j = 0; j < J; ++j) {
            float *d = out + ((size_t)base * J + j) * 12;
            for (uint32_t l = 0; l < live; ++l) {
                _mm256_storeu_ps(d + (size_t)l * s,     v8);
                _mm_storeu_ps   (d + (size_t)l * s + 8, v4);
            }
        }
    }
}

/* NON-TEMPORAL pure-streaming floor: write the same palette byte
 * volume over [begin,end) as per-instance CONTIGUOUS runs with streaming stores (no read-for-ownership) -
 * the NT write ceiling, in the SAME 16B-lead-in / 32B-middle / 16B-tail order as
 * mrw_stream_run. (temporal floor_stream − this) isolates the RFO penalty a normal store pays here; it is
 * the decisive number for whether NT can help the write at all on this µarch, independent of the kernel's
 * staging cost. One _mm_sfence() per call mirrors the kernel's per-call ordering. */
static void floor_stream_nt(float *out, uint32_t begin, uint32_t end, uint32_t J) {
    const __m256 v8 = _mm256_set1_ps(1.0f);
    const __m128 v4 = _mm_set1_ps(1.0f);
    const size_t n = (size_t)J * 12;                  /* floats per instance (16B-aligned, multiple of 16B) */
    for (uint32_t i = begin; i < end; ++i) {
        float *d = out + (size_t)i * n;
        size_t k = 0;
        if (((uintptr_t)d & 31u) != 0u) { _mm_stream_ps(d, v4); k = 4; }
        for (; k + 8 <= n; k += 8) _mm256_stream_ps(d + k, v8);
        if (k + 4 <= n) _mm_stream_ps(d + k, v4);
    }
    _mm_sfence();
}

/* pooled drivers (same per-worker chunking + scratch slicing as marrow_job). */
struct FloorNSCtx { const mrw_skeleton_view *skel; const mrw_clip_view *clip; const float *times;
                    float *out; void *scratch; size_t unit; uint32_t J; };
static void floor_ns_job(void *v, uint32_t worker, uint32_t begin, uint32_t end) {
    FloorNSCtx *c = (FloorNSCtx *)v; if (end <= begin) return;
    size_t off = (size_t)begin * c->J * 12u;
    void *scr = (char *)c->scratch + (size_t)worker * c->unit;
    mrw_floor_noscatter_avx2_fma(c->skel, c->clip, &c->times[begin], end - begin, c->out + off, MRW_PALETTE_F32, 0, (float *)scr);
}
struct FloorStreamCtx { float *out; uint32_t J; };
static void floor_stream_job(void *v, uint32_t, uint32_t begin, uint32_t end) {
    FloorStreamCtx *c = (FloorStreamCtx *)v; floor_stream(c->out, begin, end, c->J);
}
static void floor_stream_nt_job(void *v, uint32_t, uint32_t begin, uint32_t end) {
    FloorStreamCtx *c = (FloorStreamCtx *)v; floor_stream_nt(c->out, begin, end, c->J);
}

/* ====================== f16 output palette ======================
 * The f16 (binary16) output twin of the timed marrow path + its pure-streaming write floor. The palette
 * is the SAME canonical 3×4 AoS layout, narrowed to 24 B/joint - half the f32 write. This is the lever on
 * the heavy-rig output-write wall: at 16 threads compute is fully hidden behind the write, so halving the
 * bytes ~halves the pooled time toward the compute floor. */
struct MarrowJobCtxF16 {
    const mrw_dispatch *disp; const mrw_skeleton_view *skel; const mrw_clip_view *clip;
    const float *times; uint16_t *out; size_t out_bytes; void *scratch; size_t unit; uint32_t J;
};
static void marrow_job_f16(void *v, uint32_t worker, uint32_t begin, uint32_t end) {
    MarrowJobCtxF16 *c = (MarrowJobCtxF16 *)v; if (end <= begin) return;
    size_t off = (size_t)begin * c->J * 12u;                       /* uint16 elements */
    void *scr = (char *)c->scratch + (size_t)worker * c->unit;
    mrw_batch_clip_to_palette_f16(c->disp, c->skel, c->clip, &c->times[begin], end - begin,
                                  c->out + off, c->out_bytes - off * sizeof(uint16_t), scr, c->unit);
}

/* f16 pure-streaming floor: write the 24 B/joint palette in the same (tile-of-8, joint, lane) scatter
 * order/stride as the kernel - the f16 DRAM-write ceiling (half the f32 byte volume), no pose math. */
static void floor_stream_f16(uint16_t *out, uint32_t begin, uint32_t end, uint32_t J) {
    const __m128i v8 = _mm_set1_epi16((short)0x3c00);     /* 8 × f16(1.0) = 16 B */
    const size_t s = (size_t)J * 12;                       /* per-instance stride (uint16) */
    for (uint32_t base = begin; base < end; base += 8) {
        uint32_t live = end - base; if (live > 8) live = 8;
        for (uint32_t j = 0; j < J; ++j) {
            uint16_t *d = out + ((size_t)base * J + j) * 12;
            for (uint32_t l = 0; l < live; ++l) {
                /* intrinsic stores are type-punning-safe (alias any type); a raw uint64_t* cast into the
                 * uint16_t palette would be a strict-aliasing violation. v8 low 64b = 4 × f16(1.0). */
                _mm_storeu_si128((__m128i *)(void *)(d + (size_t)l * s),     v8);   /* f16[0..7]  */
                _mm_storel_epi64((__m128i *)(void *)(d + (size_t)l * s + 8), v8);   /* f16[8..11] */
            }
        }
    }
}
struct FloorStreamF16Ctx { uint16_t *out; uint32_t J; };
static void floor_stream_f16_job(void *v, uint32_t, uint32_t begin, uint32_t end) {
    FloorStreamF16Ctx *c = (FloorStreamF16Ctx *)v; floor_stream_f16(c->out, begin, end, c->J);
}

/* Exact, portable binary16 → binary32 (subnormals included) - for the f16-vs-f32 fidelity report only
 * (not timed), so it stays free of the F16C intrinsic the non-MSVC bench path doesn't enable. */
static float f16_to_f32(uint16_t h) {
    static_assert(sizeof(float) == 4, "f16_to_f32 bit-cast assumes IEEE binary32 float");
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else { int e = -1; do { man <<= 1; ++e; } while (!(man & 0x400u)); man &= 0x3ffu; bits = sign | (uint32_t)((127 - 15 - e) << 23) | (man << 13); }
    } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

/* The whole ecological cross-check: load both libraries' own importer output, reconcile joint orders
 * by name (fail-closed), run the direct-3×4 correctness gate, then the timed N-sweep (ms/frame primary). */
static int run_gltf(const char *dir, const char *glb, bool smoke, const char *mrw_override) {
    std::printf("=== marrow vs ozz-animation - REAL-ASSET ecological cross-check (UAL1) ===\n");

    auto join = [&](const char *name) { return std::string(dir) + "/" + name; };
    const std::string mrw_path = mrw_override ? std::string(mrw_override) : join("ual1_walk.mrw");
    const std::string skel_path = join("ual1_skeleton.ozz");
    const std::string opt_path = join("ual1_walk_opt.ozz");
    const std::string dense_path = join("ual1_walk_dense.ozz");

    /* ---- marrow: load .mrw, open (full loader validation), skeleton + first CLIP section ---- */
    size_t blob_size = 0;
    uint8_t *blob = read_file_64(mrw_path.c_str(), &blob_size);
    if (!blob) return 10;
    mrw_blob b;
    if (mrw_blob_open(blob, (uint64_t)blob_size, &b) != MRW_OK) {
        std::fprintf(stderr, "[gltf] %s is not a valid .mrw blob\n", mrw_path.c_str()); fr(blob); return 10;
    }
    mrw_skeleton_view skel;
    if (mrw_blob_skeleton(&b, &skel) != MRW_OK) { std::fprintf(stderr, "[gltf] .mrw has no skeleton\n"); fr(blob); return 10; }
    mrw_clip_view clip; bool have_clip = false;
    for (uint32_t si = 0; ; ++si) {
        mrw_result rc = mrw_clip_view_at(&b, si, &clip);
        if (rc == MRW_OK) { have_clip = true; break; }    /* first section that IS a clip */
        if (rc == MRW_E_RANGE) break;                     /* scanned past the last section */
        /* MRW_E_INCOMPATIBLE: section si is some other type - keep scanning */
    }
    if (!have_clip) { std::fprintf(stderr, "[gltf] .mrw has no CLIP section\n"); fr(blob); return 10; }
    const uint32_t J = skel.joint_count;                  /* marrow skin joints (expect 65) */
    if (clip.joint_count != J) { std::fprintf(stderr, "[gltf] clip/skel joint mismatch %u vs %u\n", clip.joint_count, J); fr(blob); return 10; }
    if (clip.sample_count < 2 || !(clip.fps > 0.0f)) {
        std::fprintf(stderr, "[gltf] clip needs >=2 samples and fps>0 (got %u samples, %.3f Hz)\n", clip.sample_count, clip.fps);
        fr(blob); return 10;
    }
    const float marrow_dur = (float)(clip.sample_count - 1) / clip.fps;

    /* ---- ozz: load its own skeleton + both animation variants (its real gltf2ozz output) ---- */
    ozz::animation::Skeleton ozzskel;
    if (!ozz_load_skeleton(skel_path.c_str(), &ozzskel)) { fr(blob); return 11; }
    ozz::animation::Animation anim_opt, anim_dense;
    if (!ozz_load_animation(opt_path.c_str(), &anim_opt))     { fr(blob); return 11; }
    if (!ozz_load_animation(dense_path.c_str(), &anim_dense)) { fr(blob); return 11; }
    const int ozzJ = ozzskel.num_joints();                /* MEASURED (expect 67 = 65 + Armature + Mannequin) */
    const int ozz_soa = ozzskel.num_soa_joints();
    const float ozz_dur = anim_opt.duration();
    if (anim_opt.num_tracks() != ozzJ || anim_dense.num_tracks() != ozzJ) {
        std::fprintf(stderr, "[gltf] ozz anim tracks (%d/%d) != skeleton joints %d\n",
                     anim_opt.num_tracks(), anim_dense.num_tracks(), ozzJ); fr(blob); return 11;
    }
    /* Fail closed on a duration mismatch: the exact-key structural gate and the shared time base both
     * assume both pipelines span the same [0,dur). A mismatch would silently invalidate the gate. */
    if (std::fabs(ozz_dur - marrow_dur) > 1e-3f) {
        std::fprintf(stderr, "[gltf] duration mismatch: marrow %.5fs vs ozz %.5fs - incompatible clips\n", marrow_dur, ozz_dur);
        fr(blob); return 11;
    }

    /* ---- glTF: parse for the skin inverse-bind matrices (cgltf; ozz discards them) ---- */
    cgltf_options copt; std::memset(&copt, 0, sizeof copt);
    cgltf_data *gd = nullptr;
    if (cgltf_parse_file(&copt, glb, &gd) != cgltf_result_success || !gd) {
        std::fprintf(stderr, "[gltf] cgltf parse failed: %s\n", glb); fr(blob); return 12;
    }
    if (cgltf_load_buffers(&copt, gd, glb) != cgltf_result_success) {
        std::fprintf(stderr, "[gltf] cgltf load_buffers failed (need embedded .glb buffers): %s\n", glb);
        cgltf_free(gd); fr(blob); return 12;
    }
    if (gd->skins_count != 1) {
        std::fprintf(stderr, "[gltf] expected exactly 1 skin, got %zu\n", (size_t)gd->skins_count);
        cgltf_free(gd); fr(blob); return 12;
    }
    const cgltf_skin *skin = &gd->skins[0];
    if ((uint32_t)skin->joints_count != J) {
        std::fprintf(stderr, "[gltf] skin joints %zu != marrow J %u\n", (size_t)skin->joints_count, J);
        cgltf_free(gd); fr(blob); return 12;
    }
    if (!skin->inverse_bind_matrices || skin->inverse_bind_matrices->count != skin->joints_count) {
        std::fprintf(stderr, "[gltf] skin lacks a full inverse-bind accessor (%zu of %zu)\n",
                     skin->inverse_bind_matrices ? (size_t)skin->inverse_bind_matrices->count : 0u, (size_t)skin->joints_count);
        cgltf_free(gd); fr(blob); return 12;
    }

    /* ---- skin-order → marrow-index map: the EXACT lifted DFS gltf2marrow/the demo run ---- */
    mrw_g2m_joint_order jo; char diag[256] = {0};
    if (mrw_g2m_joint_order_build(skin, &jo, diag, sizeof diag) != MRW_OK) {
        std::fprintf(stderr, "[gltf] joint-order build failed: %s\n", diag);
        cgltf_free(gd); fr(blob); return 13;
    }

    /* ---- skin-order → ozz model-joint map, BY NAME, fail-closed; IBMs in marrow order ---- */
    ozz::span<const char *const> onames = ozzskel.joint_names();
    std::vector<uint32_t> ozz_model_idx(J);               /* marrow slot mj -> ozz model joint */
    std::vector<float> ib_mo((size_t)J * 12);             /* inverse-bind in marrow-slot order */
    std::vector<ozz::math::Float4x4> ib4_mo(J);
    int fail = 0;
    for (uint32_t mj = 0; mj < J && !fail; ++mj) {
        const uint32_t s = jo.order[mj];                  /* skin-order joint feeding marrow slot mj */
        const char *nm = skin->joints[s]->name;
        if (!nm || !nm[0]) { std::fprintf(stderr, "[gltf] skin joint %u has empty name\n", s); fail = 1; break; }
        /* The .mrw must be THIS glTF's: marrow slot mj (lifted-DFS order) must carry the same joint
         * name as skin slot jo.order[mj]. A mismatch means a stale/wrong .mrw - abort, never time it. */
        const char *mn = nullptr; mrw_skeleton_joint_name(&skel, mj, &mn);
        if (!mn || std::strcmp(mn, nm) != 0) {
            std::fprintf(stderr, "[gltf] marrow joint %u name '%s' != glTF skin '%s' (stale/wrong .mrw for this glTF)\n", mj, mn ? mn : "(null)", nm);
            fail = 1; break;
        }
        int found = -1, matches = 0;
        for (int k = 0; k < ozzJ; ++k) if (onames[k] && std::strcmp(onames[k], nm) == 0) { found = k; ++matches; }
        if (matches != 1) { std::fprintf(stderr, "[gltf] joint '%s' maps to %d ozz joints (need exactly 1)\n", nm, matches); fail = 1; break; }
        ozz_model_idx[mj] = (uint32_t)found;
        float m16[16];
        if (!cgltf_accessor_read_float(skin->inverse_bind_matrices, s, m16, 16)) {
            std::fprintf(stderr, "[gltf] inverse-bind read failed at skin joint %u\n", s); fail = 1; break;
        }
        bool finite = true; for (int t = 0; t < 16; ++t) if (!std::isfinite(m16[t])) finite = false;
        /* glTF MAT4 is column-major → its bottom row is (m[3], m[7], m[11], m[15]); reject non-affine. */
        if (!finite || std::fabs(m16[3]) > 1e-4f || std::fabs(m16[7]) > 1e-4f ||
            std::fabs(m16[11]) > 1e-4f || std::fabs(m16[15] - 1.0f) > 1e-4f) {
            std::fprintf(stderr, "[gltf] non-affine/non-finite inverse-bind at skin joint %u\n", s); fail = 1; break;
        }
        g_colmajor4x4_to_affine(m16, &ib_mo[(size_t)mj * 12]);
        ib4_mo[mj] = ib_to_float4x4(&ib_mo[(size_t)mj * 12]);
        /* Verify the FAIRNESS invariant directly: marrow's baked inverse-bind (from the same glTF
         * accessor via the same conversion) must equal the one we hand ozz, bit-for-bit modulo fp.
         * This is what makes "both multiply by identical inverse-bind" a checked fact, not a claim. */
        float mib[12]; mrw_skeleton_inverse_bind(&skel, mj, mib);
        double ibd = 0.0;
        for (int t = 0; t < 12; ++t) { double d = std::fabs((double)mib[t] - (double)ib_mo[(size_t)mj * 12 + t]); if (d > ibd) ibd = d; }
        if (ibd > 1e-5) {
            std::fprintf(stderr, "[gltf] inverse-bind mismatch joint %u (max %.2e): marrow .mrw vs glTF skin differ - identical-IBM claim broken\n", mj, ibd);
            fail = 1; break;
        }
    }
    /* every skin joint must map to a DISTINCT ozz joint (catches duplicate skin names) */
    if (!fail) {
        std::vector<char> seen(ozzJ, 0);
        for (uint32_t mj = 0; mj < J; ++mj) {
            uint32_t k = ozz_model_idx[mj];
            if (seen[k]) { std::fprintf(stderr, "[gltf] two skin joints map to ozz joint %u\n", k); fail = 1; break; }
            seen[k] = 1;
        }
    }
    mrw_g2m_joint_order_free(&jo);
    cgltf_free(gd);                                       /* IBMs + ozz model map are now captured */
    if (fail) { fr(blob); return 13; }

    /* ---- assemble the ozz rig (J skin joints out of ozzJ model joints; shared time base) ---- */
    OzzRig R;
    R.skel = &ozzskel; R.anim = &anim_opt;
    R.J = (int)J; R.model_count = ozzJ; R.num_soa = ozz_soa;
    R.model_idx = ozz_model_idx.data();
    R.dur = marrow_dur;
    R.ib.assign(ib_mo.begin(), ib_mo.end());
    R.ib4 = ib4_mo;

    /* One-time validity: incompatible animation/skeleton/context sizes make ozz's jobs return false
     * (and leave the output undefined). Check once here - not per call - so the timed loop and the
     * shared procedural path stay branch-free; a failure means our load/sizing is wrong, so abort. */
    {
        OzzScratch vsc; vsc.init(ozz_soa, ozzJ);
        Contexts vcx; build_contexts(1, ozzJ, vcx);
        ozz::animation::SamplingJob sj; sj.animation = &anim_dense; sj.context = vcx.ptr[0]; sj.ratio = 0.0f;
        sj.output = ozz::span<ozz::math::SoaTransform>(vsc.locals.data(), (size_t)ozz_soa);
        bool ok = sj.Run();
        ozz::animation::LocalToModelJob l2m; l2m.skeleton = &ozzskel;
        l2m.input  = ozz::span<const ozz::math::SoaTransform>(vsc.locals.data(), (size_t)ozz_soa);
        l2m.output = ozz::span<ozz::math::Float4x4>(vsc.models.data(), (size_t)ozzJ);
        ok = l2m.Run() && ok;
        if (!ok) { std::fprintf(stderr, "[gltf] ozz Sampling/LocalToModel job rejected the loaded skeleton/animation\n"); fr(blob); return 11; }
    }

    mrw_dispatch best; mrw_dispatch_detect(&best);
    const char *best_name = best.backend == MRW_BACKEND_AVX2 ? "AVX2+FMA" : best.backend == MRW_BACKEND_SSE2 ? "SSE2" : "scalar";

    std::printf("asset: %s\n", glb);
    std::printf("  clip: marrow %u samples @ %.0f Hz (dur %.4fs) | ozz dur %.4fs\n", clip.sample_count, clip.fps, marrow_dur, ozz_dur);
    /* Codec + dense-track size: codec 0 = raw TRS (40 B/sample), codec 1 = scale-free q4+t3
     * (28 B/sample, −30%). The on-disk dense track is J × samples × stride; sweep both codecs by loading
     * a codec-0 .mrw (gltf2marrow --codec0) vs the unit-scale default codec-1 via --mrw. */
    {
        const uint32_t stride_b = (clip.codec == 1u) ? 28u : 40u;
        const size_t track_b = (size_t)J * clip.sample_count * stride_b;
        std::printf("  marrow codec %u (%s, %u B/sample) | dense track %u joints × %u samples = %zu B (%.1f KB)\n",
                    clip.codec, clip.codec == 1u ? "scale-free q4+t3" : "raw TRS q4+t3+s3",
                    stride_b, J, clip.sample_count, track_b, (double)track_b / 1024.0);
    }
    std::printf("  skin joints = %u | marrow runtime J = %u | ozz runtime J = %d  (%u skin + %d non-skin nodes, e.g. Armature + Mannequin)\n",
                J, J, ozzJ, J, ozzJ - (int)J);
    std::printf("  ozz Animation size  opt: %zu B  dense: %zu B   (dense = no key reduction; opt = ozz default optimizer)\n",
                (size_t)anim_opt.size(), (size_t)anim_dense.size());
    std::printf("  marrow best backend = %s ; ozz = AVX2 (built-in)\n", best_name);
    std::printf("  NOTE: J=%u is %.2gx the top of marrow's lean 8-20 band - a large-rig ecological STRESS case,\n"
                "        NOT evidence about lean real rigs (the lean claim is carried by the synthetic low-N crossover sweep).\n",
                J, (double)J / 20.0);

    /* ---- correctness gate + ecological-divergence report (DIRECT 3×4) ----
     * The STRUCTURAL gate (must pass sub-mm) compares marrow vs ozz-DENSE at EXACT key times - where
     * neither pipeline interpolates - so it isolates the remap + inverse-bind + 3×4 conversion from
     * any interpolation-method difference. Then we REPORT, at arbitrary between-key times, how far the
     * two full pipelines drift (ozz-dense) and how much ozz's key-reduction adds (ozz-opt): those are
     * ecological properties (different importers/interpolators), disclosed, not gated. ozz-opt stays
     * the headline (its normal shipping asset); we never tune its optimizer to chase the gate. */
    const char *headline = "ozz-opt";
    const ozz::animation::Animation *headline_anim = &anim_opt;
    {
        const uint32_t Nc = 64;
        mrw_mem_req sreq, preq;
        if (mrw_batch_clip_to_palette_requirements(J, Nc, MRW_PALETTE_F32, &sreq, &preq) != MRW_OK) { fr(blob); return 14; }
        void *scratch = al64(sreq.size);
        float *pM = (float *)al64(preq.size), *pOd = (float *)al64(preq.size), *pOo = (float *)al64(preq.size);
        float *te = (float *)al64((size_t)Nc * sizeof(float));   /* exact key times */
        float *tb = (float *)al64((size_t)Nc * sizeof(float));   /* arbitrary between-key times */
        if (!scratch || !pM || !pOd || !pOo || !te || !tb || !aligned(scratch, 64) || !aligned(pM, 16)) {
            std::fprintf(stderr, "[gltf] gate alloc/alignment failure\n"); fr(blob); return 14;
        }
        OzzScratch oss; oss.init(ozz_soa, ozzJ);
        Contexts cx; build_contexts(Nc, ozzJ, cx);
        /* Exact-key set covers keys 0..sample_count-2 (all strictly < dur). The final key sits AT
         * t==dur - the loop boundary OUTSIDE the valid [0,dur) sampling domain. There ozz's ratio wraps
         * (ratio>=1 -> ratio-1 == 0, i.e. it samples key 0) while marrow yields its terminal sample, so
         * for a clip whose key0 != key_last the two legitimately differ at that one point. Not probed. */
        const uint32_t last_key = clip.sample_count > 1 ? clip.sample_count - 1 : 1;
        for (uint32_t i = 0; i < Nc; ++i) {
            te[i] = (float)(i % last_key) / clip.fps;                    /* lands on keys 0..dur) */
            tb[i] = std::fmod((float)i * 0.0137f + 0.01f, marrow_dur);   /* mid-interval */
        }
        OzzRig Rd = R; Rd.anim = &anim_dense;
        double basis, trans, bd, td, bo, to;

        /* structural gate: exact key times, marrow vs ozz-dense */
        mrw_result grc = mrw_batch_clip_to_palette(&best, &skel, &clip, te, Nc, pM, preq.size, scratch, sreq.size);
        ozz_palette_range(Rd, cx.ptr.data(), te, 0, Nc, oss, pOd, OZZ_AFFINE3X4);
        palette_matrix_diff(pM, pOd, Nc, J, &basis, &trans);

        /* ecological divergence: between-key times, marrow vs ozz-dense and vs ozz-opt */
        grc = grc ? grc : mrw_batch_clip_to_palette(&best, &skel, &clip, tb, Nc, pM, preq.size, scratch, sreq.size);
        if (grc != MRW_OK) { std::fprintf(stderr, "[gltf] marrow batch failed in gate (%d)\n", (int)grc); fr(scratch); fr(pM); fr(pOd); fr(pOo); fr(te); fr(tb); fr(blob); return 14; }
        ozz_palette_range(Rd, cx.ptr.data(), tb, 0, Nc, oss, pOd, OZZ_AFFINE3X4);
        ozz_palette_range(R,  cx.ptr.data(), tb, 0, Nc, oss, pOo, OZZ_AFFINE3X4);
        palette_matrix_diff(pM, pOd, Nc, J, &bd, &td);
        palette_matrix_diff(pM, pOo, Nc, J, &bo, &to);

        /* Depth profile of the structural error: a correct remap/IBM gives sub-mm at the root and a
         * smooth rise with chain depth (each joint's sub-0.1° orientation diff between the two
         * importers is lever-arm-amplified downstream) - vs the constant root-level offset a units/
         * IBM/remap bug would show. We report the worst (distal) joint and the root for that contrast. */
        uint32_t worst_j = 0; double worst_mm = 0.0, root_mm = 0.0;
        for (uint32_t j = 0; j < J; ++j) {
            double w = 0.0;
            for (uint32_t i = 0; i < Nc; ++i) {
                const float *a = &pM[((size_t)i*J+j)*12], *b = &pOd[((size_t)i*J+j)*12];
                double dx=(double)a[3]-b[3], dy=(double)a[7]-b[7], dz=(double)a[11]-b[11];
                double mm = std::sqrt(dx*dx+dy*dy+dz*dz)*1000.0; if (mm>w) w=mm;
            }
            if (j == 0) root_mm = w;
            if (w > worst_mm) { worst_mm = w; worst_j = j; }
        }
        const char *worst_nm = nullptr; mrw_skeleton_joint_name(&skel, worst_j, &worst_nm);

        /* Gate (calibrated to the measured ecological floor of two independent importers):
         *  - ORIENTATION basis < 1e-3: every joint correctly oriented - the structural invariant that
         *    catches any wrong joint↔joint map or wrong inverse-bind (a mis-map is O(1) here).
         *  - TRANSLATION < 2 mm: accommodates lever-arm-amplified leaf divergence (root is sub-mm, the
         *    worst is a fingertip); still catches any gross units/IBM/remap error, which lands at cm+. */
        std::printf("\n[gate] DIRECT 3x4 palette compare over %u inst x %u joints:\n", Nc, J);
        std::printf("  STRUCTURAL  (exact key times)  marrow vs ozz-dense : basis %.6f | trans %.5f mm -> %s\n",
                    basis, trans, (trans < 2.0 && basis < 1e-3) ? "PASS" : "FAIL");
        std::printf("              depth profile: root %.4f mm  ->  worst j%u %s %.4f mm  (accumulation, not a remap defect)\n",
                    root_mm, worst_j, worst_nm ? worst_nm : "?", worst_mm);
        std::printf("  ecological  (between keys)     marrow vs ozz-dense : basis %.6f | trans %.5f mm  (interp-method drift)\n", bd, td);
        std::printf("  ecological  (between keys)     marrow vs ozz-opt   : basis %.6f | trans %.5f mm  (+ ozz key reduction)\n", bo, to);
        const bool gate_ok = (trans < 2.0) && (basis < 1e-3);
        fr(scratch); fr(pM); fr(pOd); fr(pOo); fr(te); fr(tb);
        if (!gate_ok) { std::fprintf(stderr, "[gate] STRUCTURAL correctness FAILED (marrow vs ozz-dense at key times) - aborting\n"); fr(blob); return 14; }
    }

    /* ---- Regime A N-sweep: PRIMARY ms/frame, plus ns/(i·j @ 65 skin joints) for both ---- */
    const uint32_t smoke_N[] = { 256u, 4096u };
    const uint32_t full_N[]  = { 64u, 256u, 1024u, 4096u, 16384u, 65536u };
    const uint32_t *Ns = smoke ? smoke_N : full_N;
    const int nN = smoke ? 2 : 6;
    long long ctx_bytes_big = 0;

    std::printf("\n[palette] boundary (b) - single thread, J=%u skin joints (headline = %s)\n", J, headline);
    std::printf("%8s | %11s | %11s | %11s | %10s | %10s | %7s | %9s\n",
                "N", "mrw ms/frm", "ozzO ms/frm", "ozzD ms/frm", "mrw ns/ij", "ozzO ns/ij", "spd", "p95 ms");
    for (int ni = 0; ni < nN; ++ni) {
        uint32_t N = Ns[ni];
        mrw_mem_req sreq, preq;
        if (mrw_batch_clip_to_palette_requirements(J, N, MRW_PALETTE_F32, &sreq, &preq) != MRW_OK) continue;
        void *scratch = al64(sreq.size); float *palM = (float *)al64(preq.size); float *palO = (float *)al64(preq.size);
        float *times = (float *)al64((size_t)N * sizeof(float));
        if (!scratch || !palM || !palO || !times) { std::fprintf(stderr, "[gltf] alloc fail N=%u\n", N); fr(blob); return 15; }
        OzzScratch oss; oss.init(ozz_soa, ozzJ);
        Contexts cx; build_contexts(N, ozzJ, cx);
        if (N == 16384) ctx_bytes_big = cx.bytes_per_ctx;
        for (uint32_t i = 0; i < N; ++i) times[i] = std::fmod((float)i * 0.0011f + 0.003f, marrow_dur);
        const float dt = marrow_dur / 97.0f; volatile float sink = 0;
        OzzRig Ro = R; Ro.anim = &anim_opt;
        OzzRig Rd = R; Rd.anim = &anim_dense;

        long long a0 = alloc_count();
        Stat sM = measure(N, [&] { advance_times(times, N, dt, marrow_dur);
            mrw_batch_clip_to_palette(&best, &skel, &clip, times, N, palM, preq.size, scratch, sreq.size); sink += palM[0]; }, smoke);
        Stat sO = measure(N, [&] { advance_times(times, N, dt, marrow_dur);
            ozz_palette_range(Ro, cx.ptr.data(), times, 0, N, oss, palO, OZZ_AFFINE3X4); sink += palO[0]; }, smoke);
        Stat sD = measure(N, [&] { advance_times(times, N, dt, marrow_dur);
            ozz_palette_range(Rd, cx.ptr.data(), times, 0, N, oss, palO, OZZ_AFFINE3X4); sink += palO[0]; }, smoke);
        long long inloop = alloc_count() - a0;

        std::printf("%8u | %11.4f | %11.4f | %11.4f | %10.2f | %10.2f | %6.2fx | %9.4f%s\n",
                    N, sM.mean_ms, sO.mean_ms, sD.mean_ms, ns_ij(sM.mean_ms, N, J), ns_ij(sO.mean_ms, N, J),
                    sM.mean_ms > 0 ? sO.mean_ms / sM.mean_ms : 0.0, sM.p95_ms, inloop ? "  [ALLOC!]" : "");
        if (inloop) std::fprintf(stderr, "[gate] %lld allocs in timed window N=%u\n", inloop, N);
        (void)sink; fr(scratch); fr(palM); fr(palO); fr(times);
    }
    std::printf("  (ms/frame = wall-clock for one frame's N palettes - the deliverable, primary metric.\n"
                "   ns/(i*j) normalized by %u SKIN joints for BOTH; ozz additionally samples/L2M %d-%u=%d non-skin nodes - counted as ozz overhead.)\n",
                J, ozzJ, J, ozzJ - (int)J);

    std::printf("\n[memory] ozz per-instance Context ~%lld B (ozz J=%d) => N=16384 ~%.1f MB ; marrow needs 0 (caller scratch)\n",
                ctx_bytes_big, ozzJ, (double)ctx_bytes_big * 16384.0 / (1024.0 * 1024.0));

    /* ---- Regime B (threading) at N=65536, headline ozz variant ---- */
    if (!smoke) {
        const uint32_t N = 65536;
        Jobs *pool = jobs_create(0);
        uint32_t lanes = jobs_worker_count(pool);
        mrw_mem_req sreq, preq;
        if (mrw_batch_clip_to_palette_requirements(J, N, MRW_PALETTE_F32, &sreq, &preq) != MRW_OK) { jobs_destroy(pool); fr(blob); return 16; }
        void *scratchL = al64(sreq.size * lanes); float *palM = (float *)al64(preq.size); float *palO = (float *)al64(preq.size);
        float *times = (float *)al64((size_t)N * sizeof(float));
        if (!scratchL || !palM || !palO || !times) { std::fprintf(stderr, "[gltf] Regime B alloc failure\n"); jobs_destroy(pool); fr(blob); return 16; }
        Contexts cx; build_contexts(N, ozzJ, cx);
        std::vector<OzzScratch> lanesc(lanes); for (auto &s : lanesc) s.init(ozz_soa, ozzJ);
        for (uint32_t i = 0; i < N; ++i) times[i] = std::fmod((float)i * 0.0011f + 0.003f, marrow_dur);
        const float dt = marrow_dur / 97.0f; volatile float sink = 0;
        OzzRig Rh = R; Rh.anim = headline_anim;
        MarrowJobCtx mjc{ &best, &skel, &clip, times, palM, preq.size, scratchL, sreq.size, J };
        OzzJobCtx ojc{ &Rh, cx.ptr.data(), times, palO, lanesc.data(), OZZ_AFFINE3X4 };

        Stat mS = measure(N, [&] { advance_times(times, N, dt, marrow_dur);
            mrw_batch_clip_to_palette(&best, &skel, &clip, times, N, palM, preq.size, scratchL, sreq.size); sink += palM[0]; }, false);
        Stat oS = measure(N, [&] { advance_times(times, N, dt, marrow_dur);
            ozz_palette_range(Rh, cx.ptr.data(), times, 0, N, lanesc[0], palO, OZZ_AFFINE3X4); sink += palO[0]; }, false);
        Stat mP = measure(N, [&] { advance_times(times, N, dt, marrow_dur); jobs_parallel_for(pool, N, marrow_job, &mjc); sink += palM[0]; }, false);
        Stat oP = measure(N, [&] { advance_times(times, N, dt, marrow_dur); jobs_parallel_for(pool, N, ozz_job, &ojc); sink += palO[0]; }, false);

        std::printf("\n[threading] Regime B (N=65536, J=%u, %u lanes, headline = %s)\n", J, lanes, headline);
        std::printf("  marrow serial %.4f ms/frm  pooled %.4f ms/frm  scaling %.2fx\n", mS.mean_ms, mP.mean_ms, mP.mean_ms > 0 ? mS.mean_ms / mP.mean_ms : 0.0);
        std::printf("  ozz    serial %.4f ms/frm  pooled %.4f ms/frm  scaling %.2fx\n", oS.mean_ms, oP.mean_ms, oP.mean_ms > 0 ? oS.mean_ms / oP.mean_ms : 0.0);
        std::printf("  pooled speedup marrow/ozz: %.2fx\n", mP.mean_ms > 0 ? oP.mean_ms / mP.mean_ms : 0.0);

        /* ---- synthetic floors: split the pooled time
         *      into compute vs the 204 MB output write. compute-only = real kernel minus the palette
         *      write; pure-streaming = the palette write with no math. ---- */
        FloorNSCtx fns{ &skel, &clip, times, palM, scratchL, sreq.size, J };
        FloorStreamCtx fst{ palM, J };
        Stat cS = measure(N, [&] { advance_times(times, N, dt, marrow_dur);
            mrw_floor_noscatter_avx2_fma(&skel, &clip, times, N, palM, MRW_PALETTE_F32, 0, (float *)scratchL); sink += palM[0]; }, false);
        Stat cP = measure(N, [&] { advance_times(times, N, dt, marrow_dur);
            jobs_parallel_for(pool, N, floor_ns_job, &fns); sink += palM[0]; }, false);
        Stat wS = measure(N, [&] { floor_stream(palM, 0, N, J); sink += palM[0]; }, false);
        Stat wP = measure(N, [&] { jobs_parallel_for(pool, N, floor_stream_job, &fst); sink += palM[0]; }, false);
        Stat wSn = measure(N, [&] { floor_stream_nt(palM, 0, N, J); sink += palM[0]; }, false);
        Stat wPn = measure(N, [&] { jobs_parallel_for(pool, N, floor_stream_nt_job, &fst); sink += palM[0]; }, false);

        const double wbytes = (double)N * J * 48.0;                 /* palette bytes written / frame */
        const double GBs_full = mP.mean_ms > 0 ? wbytes / (mP.mean_ms * 1e-3) / 1e9 : 0.0;
        const double GBs_str  = wP.mean_ms > 0 ? wbytes / (wP.mean_ms * 1e-3) / 1e9 : 0.0;
        const double GBs_ntw  = wPn.mean_ms > 0 ? wbytes / (wPn.mean_ms * 1e-3) / 1e9 : 0.0;
        std::printf("\n[synthetic floors] (N=%u, J=%u, %u lanes) - bracket compute vs output write\n", N, J, lanes);
        std::printf("  full kernel       serial %8.4f  pooled %8.4f ms/frm  scaling %.2fx  (delivered write %.1f GB/s)\n",
                    mS.mean_ms, mP.mean_ms, mP.mean_ms > 0 ? mS.mean_ms / mP.mean_ms : 0.0, GBs_full);
        std::printf("  compute-only      serial %8.4f  pooled %8.4f ms/frm  scaling %.2fx  (kernel minus the 204 MB write)\n",
                    cS.mean_ms, cP.mean_ms, cP.mean_ms > 0 ? cS.mean_ms / cP.mean_ms : 0.0);
        std::printf("  pure-streaming    serial %8.4f  pooled %8.4f ms/frm  scaling %.2fx  (the write, no math; %.1f GB/s)\n",
                    wS.mean_ms, wP.mean_ms, wP.mean_ms > 0 ? wS.mean_ms / wP.mean_ms : 0.0, GBs_str);
        std::printf("  pure-streaming NT serial %8.4f  pooled %8.4f ms/frm  scaling %.2fx  (NT write, no RFO; %.1f GB/s)\n",
                    wSn.mean_ms, wPn.mean_ms, wPn.mean_ms > 0 ? wSn.mean_ms / wPn.mean_ms : 0.0, GBs_ntw);
        std::printf("  pooled bracket: compute-only %.4f ms ; write adds %.4f ms ⇒ output write ≈ %.0f%% of full pooled\n",
                    cP.mean_ms, mP.mean_ms - cP.mean_ms, mP.mean_ms > 0 ? 100.0 * (mP.mean_ms - cP.mean_ms) / mP.mean_ms : 0.0);
        std::printf("  RFO term: temporal write %.4f − NT write %.4f = %.4f ms pooled (the read-for-ownership NT removes)\n",
                    wP.mean_ms, wPn.mean_ms, wP.mean_ms - wPn.mean_ms);

        /* ---- f16 output palette (24 B/joint = half f32). Same kernel/scratch; only the final
         *      AoS narrows. The write-wall lever - at 16 threads compute hides behind the write, so the
         *      halved byte volume ~halves the pooled time toward the compute floor. ---- */
        mrw_mem_req sreq16, preq16;
        if (mrw_batch_clip_to_palette_requirements(J, N, MRW_PALETTE_F16, &sreq16, &preq16) == MRW_OK) {
            uint16_t *palM16 = (uint16_t *)al64(preq16.size);
            if (!palM16) { std::fprintf(stderr, "[gltf] f16 alloc failure\n"); jobs_destroy(pool); fr(scratchL); fr(palM); fr(palO); fr(times); fr(blob); return 16; }
            MarrowJobCtxF16 mjc16{ &best, &skel, &clip, times, palM16, preq16.size, scratchL, sreq16.size, J };
            FloorStreamF16Ctx fst16{ palM16, J };
            Stat mS16 = measure(N, [&] { advance_times(times, N, dt, marrow_dur);
                mrw_batch_clip_to_palette_f16(&best, &skel, &clip, times, N, palM16, preq16.size, scratchL, sreq16.size); sink += palM16[0]; }, false);
            Stat mP16 = measure(N, [&] { advance_times(times, N, dt, marrow_dur); jobs_parallel_for(pool, N, marrow_job_f16, &mjc16); sink += palM16[0]; }, false);
            Stat wS16 = measure(N, [&] { floor_stream_f16(palM16, 0, N, J); sink += palM16[0]; }, false);
            Stat wP16 = measure(N, [&] { jobs_parallel_for(pool, N, floor_stream_f16_job, &fst16); sink += palM16[0]; }, false);

            const double wbytes16 = (double)N * J * 24.0;
            const double GBs_f16  = mP16.mean_ms > 0 ? wbytes16 / (mP16.mean_ms * 1e-3) / 1e9 : 0.0;
            const double GBs_str16 = wP16.mean_ms > 0 ? wbytes16 / (wP16.mean_ms * 1e-3) / 1e9 : 0.0;
            std::printf("\n[f16 output palette] (N=%u, J=%u, %u lanes) - halve the write wall\n", N, J, lanes);
            std::printf("  f32 full kernel   serial %8.4f  pooled %8.4f ms/frm  scaling %.2fx  (write %.1f GB/s, %.0f MB)\n",
                        mS.mean_ms, mP.mean_ms, mP.mean_ms > 0 ? mS.mean_ms / mP.mean_ms : 0.0, GBs_full, wbytes16 * 2.0 / 1e6);
            std::printf("  f16 full kernel   serial %8.4f  pooled %8.4f ms/frm  scaling %.2fx  (write %.1f GB/s, %.0f MB)\n",
                        mS16.mean_ms, mP16.mean_ms, mP16.mean_ms > 0 ? mS16.mean_ms / mP16.mean_ms : 0.0, GBs_f16, wbytes16 / 1e6);
            std::printf("  f16/f32 pooled speedup %.2fx ; ST ns/i·j  f32 %.2f → f16 %.2f (ST is compute-bound: ≈flat)\n",
                        mP16.mean_ms > 0 ? mP.mean_ms / mP16.mean_ms : 0.0, ns_ij(mS.mean_ms, N, J), ns_ij(mS16.mean_ms, N, J));
            std::printf("  pure-streaming f16 serial %8.4f  pooled %8.4f ms/frm  scaling %.2fx  (the f16 write, no math; %.1f GB/s)\n",
                        wS16.mean_ms, wP16.mean_ms, wP16.mean_ms > 0 ? wS16.mean_ms / wP16.mean_ms : 0.0, GBs_str16);

            /* f16-vs-f32 fidelity on the REAL asset (the owner-run cross-check, palette level): max basis
             * diff and max joint-origin (translation column) distance across all N×J palette entries. */
            for (uint32_t i = 0; i < N; ++i) times[i] = std::fmod((float)i * 0.0011f + 0.003f, marrow_dur);
            mrw_batch_clip_to_palette(&best, &skel, &clip, times, N, palM, preq.size, scratchL, sreq.size);
            mrw_batch_clip_to_palette_f16(&best, &skel, &clip, times, N, palM16, preq16.size, scratchL, sreq16.size);
            static const int bidx[9] = { 0,1,2, 4,5,6, 8,9,10 };
            double f16_basis = 0.0, f16_mm = 0.0;
            for (size_t e = 0; e < (size_t)N * J; ++e) {
                const float *a = &palM[e * 12]; const uint16_t *b = &palM16[e * 12];
                for (int k = 0; k < 9; ++k) { double d = std::fabs((double)a[bidx[k]] - (double)f16_to_f32(b[bidx[k]])); if (d > f16_basis) f16_basis = d; }
                double dx = (double)a[3] - f16_to_f32(b[3]), dy = (double)a[7] - f16_to_f32(b[7]), dz = (double)a[11] - f16_to_f32(b[11]);
                double mm = std::sqrt(dx * dx + dy * dy + dz * dz) * 1000.0; if (mm > f16_mm) f16_mm = mm;
            }
            std::printf("  f16 vs f32 palette: max basis %.2e | max joint-origin %.4f mm  (real-asset f16 fidelity)\n",
                        f16_basis, f16_mm);
            fr(palM16);
        }

        (void)sink; jobs_destroy(pool); fr(scratchL); fr(palM); fr(palO); fr(times);
    }

    fr(blob);
    std::printf("\n(done - UAL1 ecological cross-check)\n");
    return 0;
}
#endif /* MRW_OZZ_BENCH_GLTF */

/* ------------------------------------------------------------------ harness */

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   /* unbuffered: progress visible while piped/redirected */
    bool smoke = false, crossover = false;
    const char *gltf_dir = nullptr, *glb_path = nullptr, *mrw_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--smoke") == 0) smoke = true;
        else if (std::strcmp(argv[i], "--crossover") == 0) crossover = true;
        else if (std::strcmp(argv[i], "--gltf") == 0 && i + 1 < argc) gltf_dir = argv[++i];
        else if (std::strcmp(argv[i], "--glb")  == 0 && i + 1 < argc) glb_path = argv[++i];
        else if (std::strcmp(argv[i], "--mrw")  == 0 && i + 1 < argc) mrw_path = argv[++i];
    }

    ozz::memory::Allocator *def = ozz::memory::default_allocator();
    static CountingAllocator counting(def);
    ozz::memory::SetDefaulAllocator(&counting);

#ifdef MRW_OZZ_BENCH_GLTF
    if (gltf_dir) {
        std::string glb = glb_path ? std::string(glb_path) : (std::string(gltf_dir) + "/UAL1_Standard_RM.glb");
        return run_gltf(gltf_dir, glb.c_str(), smoke, mrw_path);
    }
#else
    if (gltf_dir) { std::fprintf(stderr, "--gltf requires building with -DMRW_OZZ_BENCH_GLTF=ON\n"); (void)glb_path; (void)mrw_path; return 1; }
#endif

    const float fps = 30.0f; const uint32_t sc = 31;          /* 1.0s @ 30fps */
    mrw_dispatch best, scal;
    mrw_dispatch_detect(&best); mrw_dispatch_scalar(&scal);
    const char *best_name = best.backend == MRW_BACKEND_AVX2 ? "AVX2+FMA" : best.backend == MRW_BACKEND_SSE2 ? "SSE2" : "scalar";

    Assets A14;
    if (!build_assets(14, sc, fps, A14)) return 1;

    std::printf("=== marrow vs ozz-animation - CPU palette throughput ===\n");
    std::printf("rig: chain, clip %u frames @ %.0ffps (dur %.3fs);  marrow best=%s;  ozz=AVX2 (built-in)\n",
                sc, fps, A14.rig.dur, best_name);
    std::printf("metric: ns/(instance*joint), single thread unless noted; ozz palette = ozz-affine3x4 (opt anim)\n");

    /* ---- correctness gate (sub-mm) ---- */
    {
        const uint32_t Nc = 64;
        mrw_mem_req sreq, preq; mrw_batch_clip_to_palette_requirements(14, Nc, MRW_PALETTE_F32, &sreq, &preq);
        void *scratch = al64(sreq.size); float *palM = (float*)al64(preq.size); float *palO = (float*)al64(preq.size);
        float *times = (float*)al64((size_t)Nc*sizeof(float));
        OzzScratch oss; oss.init(A14.orig.num_soa, 14);
        Contexts cx; build_contexts(Nc, 14, cx);
        for (uint32_t i = 0; i < Nc; ++i) times[i] = std::fmod((float)i*0.0137f+0.01f, A14.rig.dur);
        mrw_batch_clip_to_palette(&best, &A14.mr.skel, &A14.mr.clip, times, Nc, palM, preq.size, scratch, sreq.size);
        double worst = 0; OzzMode ms[2]={OZZ_AFFINE3X4,OZZ_MUL4X4}; const char* mn[2]={"ozz-affine3x4","ozz-4x4"};
        for (int mi=0; mi<2; ++mi) {
            ozz_palette_range(A14.orig, cx.ptr.data(), times, 0, Nc, oss, palO, ms[mi]);
            double mm = palette_probe_diff(A14.rig, palM, palO, Nc);
            std::printf("  correctness marrow-batch vs %-13s : max %.5f mm -> %s\n", mn[mi], mm, mm<1.0?"PASS":"FAIL");
            if (mm>worst) worst=mm;
        }
        fr(scratch); fr(palM); fr(palO); fr(times);
        if (worst >= 1.0) { std::fprintf(stderr, "[gate] correctness FAILED - aborting\n"); return 2; }
    }

    /* ---- crossover sweep: fine low-N grid to locate the exact break-even N. marrow's batch
     * processes instances in tiles of MRW_LANES (=8); at low N the unfilled lanes are paid for, so the
     * per-instance cost falls as 1/N until a tile fills. This sweep walks that curve through ozz's
     * (flat, cache-resident) ~23 ns/i·j to pin the integer N where marrow overtakes. ---- */
    if (crossover) {
        const uint32_t Nx[] = { 1u,2u,3u,4u,5u,6u,7u,8u,10u,12u,16u,24u,32u,48u,64u,96u,128u,256u };
        const int nX = (int)(sizeof(Nx)/sizeof(Nx[0]));
        std::printf("\n[crossover] boundary (b) PALETTE - fine low-N sweep, J=14 (mrw=%s vs ozz3x4-opt)\n", best_name);
        std::printf("%6s | %9s | %9s | %8s\n", "N", "mrw_best", "ozz3x4o", "speedup");
        for (int ni = 0; ni < nX; ++ni) {
            uint32_t N = Nx[ni];
            mrw_mem_req sreq, preq;
            if (mrw_batch_clip_to_palette_requirements(14, N, MRW_PALETTE_F32, &sreq, &preq) != MRW_OK) continue;
            void *scratch=al64(sreq.size); float *palM=(float*)al64(preq.size); float *palO=(float*)al64(preq.size);
            float *times=(float*)al64((size_t)N*sizeof(float));
            OzzScratch oss; oss.init(A14.orig.num_soa, 14);
            Contexts cx; build_contexts(N, 14, cx);
            for (uint32_t i=0;i<N;++i) times[i]=std::fmod((float)i*0.0011f+0.003f, A14.rig.dur);
            const float dt=A14.rig.dur/97.0f, dur=A14.rig.dur; volatile float sink=0;
            OzzRig R=A14.orig;
            Stat sB = measure(N, [&]{ advance_times(times,N,dt,dur);
                mrw_batch_clip_to_palette(&best,&A14.mr.skel,&A14.mr.clip,times,N,palM,preq.size,scratch,sreq.size); sink+=palM[0]; }, false);
            R.anim=A14.anim_opt.get();
            Stat o3 = measure(N, [&]{ advance_times(times,N,dt,dur);
                ozz_palette_range(R,cx.ptr.data(),times,0,N,oss,palO,OZZ_AFFINE3X4); sink+=palO[0]; }, false);
            std::printf("%6u | %9.2f | %9.2f | %7.2fx\n", N, ns_ij(sB.mean_ms,N,14), ns_ij(o3.mean_ms,N,14),
                        sB.mean_ms>0?o3.mean_ms/sB.mean_ms:0.0);
            (void)sink; fr(scratch); fr(palM); fr(palO); fr(times);
        }
        fr(A14.mr.blob);
        std::printf("\n(done)\n");
        return 0;
    }

    const uint32_t smoke_N[] = { 256u, 4096u };
    const uint32_t full_N[]  = { 1u, 64u, 256u, 1024u, 4096u, 16381u, 16384u, 65536u };
    const uint32_t *Ns = smoke ? smoke_N : full_N;
    int nN = smoke ? 2 : 8;
    long long ctx_bytes_16k = 0;

    /* ============ Table 1: boundary (b) PALETTE, N-sweep, J=14, Regime A ============ */
    std::printf("\n[Table 1] boundary (b) PALETTE - single thread, J=14\n");
    std::printf("%8s | %9s | %9s | %9s | %9s | %9s | %8s | %8s\n",
                "N", "mrw_best", "mrw_scal", "ozz3x4o", "ozz4x4o", "ozz3x4d", "spd_best", "p95_mrw");
    for (int ni = 0; ni < nN; ++ni) {
        uint32_t N = Ns[ni];
        mrw_mem_req sreq, preq;
        if (mrw_batch_clip_to_palette_requirements(14, N, MRW_PALETTE_F32, &sreq, &preq) != MRW_OK) continue;
        void *scratch = al64(sreq.size); float *palM=(float*)al64(preq.size); float *palO=(float*)al64(preq.size);
        float *times=(float*)al64((size_t)N*sizeof(float));
        if (!scratch||!palM||!palO||!times||!aligned(scratch,64)||!aligned(palM,16)) { std::fprintf(stderr,"alloc/align N=%u\n",N); return 3; }
        OzzScratch oss; oss.init(A14.orig.num_soa, 14);
        Contexts cx; build_contexts(N, 14, cx);
        if (N == 16384) ctx_bytes_16k = cx.bytes_per_ctx;
        for (uint32_t i=0;i<N;++i) times[i]=std::fmod((float)i*0.0011f+0.003f, A14.rig.dur);
        const float dt = A14.rig.dur/97.0f, dur = A14.rig.dur; volatile float sink=0;
        OzzRig R = A14.orig;   /* persistent; R.anim set per measure (no per-call copy) */
        auto run_mrw=[&](const mrw_dispatch* d){ return measure(N, [&, d]{ advance_times(times,N,dt,dur);
            mrw_batch_clip_to_palette(d,&A14.mr.skel,&A14.mr.clip,times,N,palM,preq.size,scratch,sreq.size); sink+=palM[0]; }, smoke); };
        auto run_ozz=[&](const ozz::animation::Animation* an, OzzMode mode){ R.anim=an; return measure(N, [&, mode]{
            advance_times(times,N,dt,dur);
            ozz_palette_range(R,cx.ptr.data(),times,0,N,oss,palO,mode); sink+=palO[0]; }, smoke); };

        long long a0 = alloc_count();
        Stat sB = run_mrw(&best);
        Stat sS = run_mrw(&scal);
        Stat o3o = run_ozz(A14.anim_opt.get(),  OZZ_AFFINE3X4);
        Stat o4o = run_ozz(A14.anim_opt.get(),  OZZ_MUL4X4);
        Stat o3d = run_ozz(A14.anim_dense.get(), OZZ_AFFINE3X4);
        long long inloop = alloc_count() - a0;

        std::printf("%8u | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f | %7.2fx | %8.2f%s\n",
                    N, ns_ij(sB.mean_ms,N,14), ns_ij(sS.mean_ms,N,14), ns_ij(o3o.mean_ms,N,14),
                    ns_ij(o4o.mean_ms,N,14), ns_ij(o3d.mean_ms,N,14),
                    sB.mean_ms>0?o3o.mean_ms/sB.mean_ms:0.0, ns_ij(sB.p95_ms,N,14),
                    inloop?"  [ALLOC!]":"");
        if (inloop) std::fprintf(stderr,"[gate] %lld allocs in timed window N=%u\n", inloop, N);
        (void)sink;
        fr(scratch); fr(palM); fr(palO); fr(times);
    }

    /* ============ Table 2: boundary (a) decomposition (ozz) + marrow batch/loop, N=16384 J=14 ====== */
    {
        const uint32_t N = 16384;
        mrw_mem_req sreq, preq; mrw_batch_clip_to_palette_requirements(14, N, MRW_PALETTE_F32, &sreq, &preq);
        void *scratch=al64(sreq.size); float *palM=(float*)al64(preq.size); float *palO=(float*)al64(preq.size);
        float *times=(float*)al64((size_t)N*sizeof(float));
        OzzScratch oss; oss.init(A14.orig.num_soa,14);
        Contexts cx; build_contexts(N,14,cx);
        for (uint32_t i=0;i<N;++i) times[i]=std::fmod((float)i*0.0011f+0.003f, A14.rig.dur);
        const float dt=A14.rig.dur/97.0f; volatile float sink=0; OzzRig R=A14.orig;
        auto ozz_fn=[&](OzzMode mode){ R.anim=A14.anim_opt.get(); advance_times(times,N,dt,A14.rig.dur);
            ozz_palette_range(R,cx.ptr.data(),times,0,N,oss,palO,mode); sink+=palO[0]; };
        Stat oa = measure(N,[&]{ozz_fn(OZZ_MODEL);}, smoke);
        Stat ob = measure(N,[&]{ozz_fn(OZZ_AFFINE3X4);}, smoke);
        std::printf("\n[Table 2] boundary (a) MODEL decomposition (ozz, N=16384, J=14)\n");
        std::printf("  ozz(a) model %.2f ns/ij | ozz(b) palette %.2f ns/ij | ozz(b)-ozz(a) = %.2f ns/ij (fused inverse-bind marrow folds in)\n",
                    ns_ij(oa.mean_ms,N,14), ns_ij(ob.mean_ms,N,14), ns_ij(ob.mean_ms-oa.mean_ms,N,14));

        /* marrow-only: batch SIMD vs loop of single-instance scalar oracle */
        float *loopscr=(float*)al64((size_t)14*12*sizeof(float));
        Stat mb = measure(N,[&]{ advance_times(times,N,dt,A14.rig.dur);
            mrw_batch_clip_to_palette(&best,&A14.mr.skel,&A14.mr.clip,times,N,palM,preq.size,scratch,sreq.size); sink+=palM[0]; }, smoke);
        Stat ml = measure(N,[&]{ advance_times(times,N,dt,A14.rig.dur);
            for (uint32_t i=0;i<N;++i) mrw_clip_to_palette(&A14.mr.skel,&A14.mr.clip,times[i],loopscr,&palM[(size_t)i*14*12],14); sink+=palM[0]; }, smoke);
        std::printf("[Table 3] marrow-only: batch SIMD %.2f ns/ij | looped scalar oracle %.2f ns/ij | batch speedup %.2fx\n",
                    ns_ij(mb.mean_ms,N,14), ns_ij(ml.mean_ms,N,14), mb.mean_ms>0?ml.mean_ms/mb.mean_ms:0.0);
        (void)sink;
        fr(loopscr); fr(scratch); fr(palM); fr(palO); fr(times);
    }

    /* ============ Table 4: joint-count crossover at N=16384, Regime A ============ */
    std::printf("\n[Table 4] joint-count crossover (N=16384, boundary b, opt)\n");
    std::printf("%4s | %9s | %9s | %8s\n", "J", "mrw_best", "ozz3x4o", "speedup");
    {
        const uint32_t N = smoke ? 1024 : 16384;
        uint32_t Js[3] = { 8, 14, 20 };
        for (int ji=0; ji<3; ++ji) {
            uint32_t J = Js[ji];
            Assets *Ap; Assets tmp;
            if (J==14) Ap=&A14; else { if(!build_assets(J,sc,fps,tmp)) return 1; Ap=&tmp; }
            mrw_mem_req sreq,preq; mrw_batch_clip_to_palette_requirements(J,N,MRW_PALETTE_F32,&sreq,&preq);
            void *scratch=al64(sreq.size); float *palM=(float*)al64(preq.size); float *palO=(float*)al64(preq.size);
            float *times=(float*)al64((size_t)N*sizeof(float));
            OzzScratch oss; oss.init(Ap->orig.num_soa,(int)J);
            Contexts cx; build_contexts(N,(int)J,cx);
            for (uint32_t i=0;i<N;++i) times[i]=std::fmod((float)i*0.0011f+0.003f, Ap->rig.dur);
            const float dt=Ap->rig.dur/97.0f; volatile float sink=0; OzzRig R=Ap->orig;
            Stat sB=measure(N,[&]{ advance_times(times,N,dt,Ap->rig.dur);
                mrw_batch_clip_to_palette(&best,&Ap->mr.skel,&Ap->mr.clip,times,N,palM,preq.size,scratch,sreq.size); sink+=palM[0]; }, smoke);
            Stat o3=measure(N,[&]{ R.anim=Ap->anim_opt.get(); advance_times(times,N,dt,Ap->rig.dur);
                ozz_palette_range(R,cx.ptr.data(),times,0,N,oss,palO,OZZ_AFFINE3X4); sink+=palO[0]; }, smoke);
            std::printf("%4u | %9.2f | %9.2f | %7.2fx\n", J, ns_ij(sB.mean_ms,N,J), ns_ij(o3.mean_ms,N,J),
                        sB.mean_ms>0?o3.mean_ms/sB.mean_ms:0.0);
            (void)sink; fr(scratch); fr(palM); fr(palO); fr(times);
        }
    }

    /* ============ Table 5: clip memory & keyframes & context bytes (J=14) ============ */
    {
        size_t mrw_clip_bytes = (size_t)14 * sc * 10 * sizeof(float);   /* dense sample array */
        std::printf("\n[Table 5] memory (J=14)\n");
        std::printf("  marrow clip (dense codec-0 samples) : %zu B   (%zu B/frame)\n",
                    mrw_clip_bytes, mrw_clip_bytes / sc);
        std::printf("  ozz Animation size  opt: %zu B (keys %zu)  dense: %zu B (keys %zu)\n",
                    A14.anim_opt->size(), A14.keys_opt, A14.anim_dense->size(), A14.keys_dense);
        std::printf("  ozz per-instance Context: ~%lld B  => N=16384 contexts ~%.1f MB (marrow needs 0)\n",
                    ctx_bytes_16k, (double)ctx_bytes_16k * 16384.0 / (1024.0*1024.0));
    }

    /* ============ Table 6: Regime B (threading) at N=65536, J=14 ============ */
    if (!smoke) {
        const uint32_t N = 65536;
        Jobs *pool = jobs_create(0);
        uint32_t lanes = jobs_worker_count(pool);
        mrw_mem_req sreq, preq; mrw_batch_clip_to_palette_requirements(14, N, MRW_PALETTE_F32, &sreq, &preq);
        void *scratchL = al64(sreq.size * lanes); float *palM=(float*)al64(preq.size); float *palO=(float*)al64(preq.size);
        float *times=(float*)al64((size_t)N*sizeof(float));
        Contexts cx; build_contexts(N,14,cx);
        std::vector<OzzScratch> lanesc(lanes); for (auto &s : lanesc) s.init(A14.orig.num_soa,14);
        for (uint32_t i=0;i<N;++i) times[i]=std::fmod((float)i*0.0011f+0.003f, A14.rig.dur);
        const float dt=A14.rig.dur/97.0f; volatile float sink=0; OzzRig R=A14.orig; R.anim=A14.anim_opt.get();

        MarrowJobCtx mjc{ &best,&A14.mr.skel,&A14.mr.clip,times,palM,preq.size,scratchL,sreq.size,14 };
        OzzJobCtx ojc{ &R, cx.ptr.data(), times, palO, lanesc.data(), OZZ_AFFINE3X4 };

        /* serial (1 lane) baselines */
        Stat mS = measure(N,[&]{ advance_times(times,N,dt,A14.rig.dur);
            mrw_batch_clip_to_palette(&best,&A14.mr.skel,&A14.mr.clip,times,N,palM,preq.size,scratchL,sreq.size); sink+=palM[0]; }, false);
        Stat oS = measure(N,[&]{ advance_times(times,N,dt,A14.rig.dur);
            ozz_palette_range(R,cx.ptr.data(),times,0,N,lanesc[0],palO,OZZ_AFFINE3X4); sink+=palO[0]; }, false);
        /* pooled */
        Stat mP = measure(N,[&]{ advance_times(times,N,dt,A14.rig.dur); jobs_parallel_for(pool,N,marrow_job,&mjc); sink+=palM[0]; }, false);
        Stat oP = measure(N,[&]{ advance_times(times,N,dt,A14.rig.dur); jobs_parallel_for(pool,N,ozz_job,&ojc); sink+=palO[0]; }, false);

        std::printf("\n[Table 6] Regime B threading (N=65536, J=14, %u lanes)\n", lanes);
        std::printf("  marrow  serial %.4f ms/frm  pooled %.4f ms/frm  scaling %.2fx\n",
                    mS.mean_ms, mP.mean_ms, mP.mean_ms>0?mS.mean_ms/mP.mean_ms:0.0);
        std::printf("  ozz3x4  serial %.4f ms/frm  pooled %.4f ms/frm  scaling %.2fx\n",
                    oS.mean_ms, oP.mean_ms, oP.mean_ms>0?oS.mean_ms/oP.mean_ms:0.0);
        std::printf("  pooled speedup marrow/ozz: %.2fx\n", mP.mean_ms>0?oP.mean_ms/mP.mean_ms:0.0);
        (void)sink; jobs_destroy(pool);
        fr(scratchL); fr(palM); fr(palO); fr(times);
    }

    fr(A14.mr.blob);
    std::printf("\n(done)\n");
    return 0;
}
