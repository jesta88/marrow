/* Tiny linear-algebra helpers for the demo (column-major mat4, Vulkan clip space).
 * The marrow runtime needs none of this - it emits row-major 3x4 affines; this is purely
 * the demo's camera/projection plumbing. Column-major to match GLSL/SPIR-V std140/430.
 *
 * Vulkan clip space: NDC depth is [0,1] and +Y points DOWN. mat4_perspective_vk bakes the
 * Y flip in (m11 negated) and maps depth to [0,1], so the rest of the demo can ignore it. */
#ifndef DEMO_LINALG_H
#define DEMO_LINALG_H

#include <math.h>

typedef struct { float x, y, z; } vec3;
typedef struct { float m[16]; } mat4; /* column-major: m[col*4 + row] */

static inline vec3 v3(float x, float y, float z) { vec3 r = { x, y, z }; return r; }
static inline vec3 v3_add(vec3 a, vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline vec3 v3_sub(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline vec3 v3_scale(vec3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline float v3_dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline vec3 v3_cross(vec3 a, vec3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline vec3 v3_normalize(vec3 a) {
    float n = sqrtf(v3_dot(a, a));
    return n > 0.0f ? v3_scale(a, 1.0f / n) : a;
}

static inline mat4 mat4_identity(void) {
    mat4 r = { { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 } };
    return r;
}

/* out = a * b (column-major; applies b first to a column vector). */
static inline mat4 mat4_mul(mat4 a, mat4 b) {
    mat4 r;
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    return r;
}

static inline mat4 mat4_translate(vec3 t) {
    mat4 r = mat4_identity();
    r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
    return r;
}

static inline mat4 mat4_scale(vec3 s) {
    mat4 r = mat4_identity();
    r.m[0] = s.x; r.m[5] = s.y; r.m[10] = s.z;
    return r;
}

static inline mat4 mat4_rotate_y(float a) {
    float c = cosf(a), s = sinf(a);
    mat4 r = mat4_identity();
    r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c;
    return r;
}

static inline mat4 mat4_rotate_x(float a) {
    float c = cosf(a), s = sinf(a);
    mat4 r = mat4_identity();
    r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c;
    return r;
}

/* Right-handed look-at (eye looks toward center, +up). */
static inline mat4 mat4_look_at(vec3 eye, vec3 center, vec3 up) {
    vec3 f = v3_normalize(v3_sub(center, eye));
    vec3 s = v3_normalize(v3_cross(f, up));
    vec3 u = v3_cross(s, f);
    mat4 r = mat4_identity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8]  = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9]  = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -v3_dot(s, eye);
    r.m[13] = -v3_dot(u, eye);
    r.m[14] =  v3_dot(f, eye);
    return r;
}

/* Right-handed perspective for Vulkan: depth [0,1], +Y-down (m11 negated). */
static inline mat4 mat4_perspective_vk(float fovy_rad, float aspect, float znear, float zfar) {
    float t = tanf(fovy_rad * 0.5f);
    mat4 r = { { 0 } };
    r.m[0]  = 1.0f / (aspect * t);
    r.m[5]  = -1.0f / t;                 /* Vulkan Y points down */
    r.m[10] = zfar / (znear - zfar);
    r.m[11] = -1.0f;
    r.m[14] = (znear * zfar) / (znear - zfar);
    return r;
}

#endif /* DEMO_LINALG_H */
