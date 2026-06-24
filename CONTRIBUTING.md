# Contributing to marrow

Thanks for your interest in marrow. This document covers how to build, test, and submit changes.

marrow is a pure-C11, zero-dependency animation runtime. The public API — and the contracts that govern
it (conventions, ABI, the `.mrw` wire format) — is the single header [`marrow.h`](marrow.h), documented
inline. Read it before making a non-trivial change.

## Building and testing

marrow builds with CMake (Ninja recommended). The runtime has no third-party dependencies.

```sh
cmake -S . -B out/build -G Ninja
cmake --build out/build
ctest --test-dir out/build --output-on-failure
```

On Windows with Visual Studio, use the supplied preset:

```sh
cmake --preset x64-release
cmake --build out/build/x64-release
ctest --test-dir out/build/x64-release --output-on-failure
```

Useful options: `MRW_BUILD_TOOLS` (gltf2marrow, marrow-bake), `MRW_BUILD_DEMO` (Vulkan demo),
`MRW_BUILD_OZZ_BENCH` (throughput comparison), `MRW_SANITIZE=address,undefined` (sanitizer build),
`MRW_BUILD_FUZZ` (libFuzzer loader target, Clang only). See [`README.md`](README.md) for the full table.

Keep the full CTest suite green on at least one toolchain before opening a PR; CI then runs it across
GCC / Clang (ASan+UBSan, MSan) / MSVC plus the loader fuzzer.

## Conventions

- **Symbol prefix.** All C symbols use `mrw_` (types, functions) / `MRW_` (macros, enums) — e.g.
  `mrw_clip_view`, `MRW_OK`. The public header is `marrow.h`; the on-disk format is `.mrw` (magic
  `MRRW`). Internal source filenames keep the `marrow_` stem (`src/marrow_pose.c`).
- **No runtime dependencies, no allocation.** The runtime is pure C11 with no libc beyond `<math.h>`,
  allocates nothing, touches no graphics API, and spawns no thread it doesn't own. Buffers are
  caller-owned and sized by `*_requirements()` queries (size **and** alignment). The banned-construct
  scan (a CTest) enforces the no-malloc/alloca/VLA/threading rule on the runtime sources.
- **ABI hygiene.** Outputs go through out-params plus a `mrw_result` return code; no nontrivial structs
  returned by value; fixed-width unsigned counts; no compiler SIMD types in public structs; the header
  compiles as both C11 and C++.
- **SIMD dispatch.** Scalar, SSE2, and AVX2 (+FMA, +F16C) kernels live in separately-flagged
  translation units, selected through a small caller-owned, immutable dispatch value. The **scalar
  backend is the reference** the SIMD kernels are checked against — keep them equivalent (FMA and
  reordered reductions are visual-only, not bit-identical). The ISA scan (a CTest) asserts each TU
  stays in its lane.
- **The loader validates, then views.** `.mrw` is the only untrusted-input boundary. New loader code
  must fail cleanly on truncated/corrupt input and stay ASan/UBSan/MSan clean; add corpus seeds for
  any new loader shape.
- **Wire-format changes are backward-additive.** A new on-disk layout takes a **new** `codec` /
  `encoding` / section-type id; existing ids are never reinterpreted, so old assets keep loading.

## Comments and documentation

Explain *why*, not *what*, and write for an outside reader of the shipped library — a user of
`marrow.h` or a new contributor to `src/`. Comments must be **self-contained**: state the rule inline
rather than citing an internal document the reader won't have. Keep statements stable and present-tense
so they don't rot.

## Submitting changes

1. Branch from `master`.
2. Keep changes focused; match the surrounding code's style and comment density.
3. Add or update tests for the behavior you change — new SIMD paths need scalar-parity coverage; new
   loader paths need malformed-input and corpus coverage.
4. Run the full CTest suite locally.
5. Open a pull request describing the change and how you verified it.

## License

By contributing, you agree that your contributions are licensed under the project's
[MIT License](LICENSE).
