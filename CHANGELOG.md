# Changelog

All notable changes to marrow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While marrow is pre-1.0, the public API
and the `.mrw` wire format may still change between minor versions; on-disk layouts always evolve
backward-additively (existing assets keep loading).

## [0.1.0] — 2026-06-24

First public release.

### Added
- **CPU runtime (Tier A):** math core, runtime SIMD dispatch (scalar / SSE2 / AVX2 +FMA +F16C), the
  `.mrw` validating loader, clip sampling, the canonical 3×4 skinning palette, pose algebra (blend /
  make-additive / accumulate / per-joint mask), analytic two-bone and aim IK, and root motion.
- **Across-instance batch path:** `mrw_batch_clip_to_palette` and the blend/additive batch variants,
  vectorized across instances (lane *i* = instance *i*), with an opt-in f16 output palette.
- **Baked GPU crowd tier (Tier B):** the baked-texture section format and a runtime consumer/validator,
  with reference GLSL + HLSL decode shaders in `examples/skinning/`.
- **Offline tools:** `gltf2marrow` (glTF 2.0 import) and `marrow-bake` (Tier-B baking with
  decomposability validation; ineligible rigs are rejected, not silently approximated).
- **`.mrw` wire format v0:** little-endian, byte-defined, with clip codecs 0 (raw fixed-rate TRS) and
  1 (scale-free q4+t3) and baked encoding 1 (quat+T+uniform-scale).
- **Verification:** randomized scalar-vs-SIMD parity, golden byte fixtures, malformed-input tests, a
  coverage-guided loader fuzzer with a portable corpus replayer, ISA / banned-construct scans, and CI
  across GCC / Clang (ASan+UBSan, MSan) / MSVC.

[0.1.0]: https://github.com/jesta88/marrow/releases/tag/v0.1.0
