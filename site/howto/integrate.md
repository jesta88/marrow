Integrate marrow into your build {#howto_integrate}
================================

marrow is engine- and build-system-agnostic. The runtime is the C sources under `src/` plus the
single public header `marrow.h` — no third-party dependencies, nothing to install.

[TOC]

## The one rule that matters: per-translation-unit ISA flags

marrow ships scalar, SSE2, and AVX2 (+FMA, +F16C) implementations of the batch kernels in
**separate `.c` files**, and picks between them at runtime (see @ref howto_crowd). For that to be
correct, **each kernel TU must be compiled with exactly its own ISA — and no more.**

> **Do not compile the whole runtime with `-mavx2`, `-march=native`, or `/arch:AVX2`.**
> If you do, the compiler may emit AVX instructions into the SSE2 baseline TU, which then crashes
> with an illegal-instruction fault on any CPU that lacks AVX — exactly the fallback path that
> exists to keep those CPUs working.

Compile the bulk of the runtime at your project's normal baseline (on x64, SSE2). Apply per-file
flags only to the kernel TUs:

| Translation unit(s) | GCC / Clang | MSVC |
|---|---|---|
| `marrow_batch_sse2.c`, `marrow_blend_sse2.c` | `-msse2 -mno-avx -mno-avx2 -mno-fma -mno-f16c` | *(baseline; no flag)* |
| `marrow_batch_avx2.c`, `marrow_blend_avx2.c` | `-mavx2 -mf16c -mno-fma -ffp-contract=off` | `/arch:AVX2` |
| `marrow_batch_avx2_fma.c`, `marrow_blend_avx2_fma.c` | `-mavx2 -mfma -mf16c` | `/arch:AVX2` |

The `-mno-*` flags on the SSE2 TU are the guardrail: appended *after* your global flags, they win,
so a project-wide `-mavx`/`-march=native` cannot leak AVX into the baseline. On MSVC x64 there is
no per-file way to lower below the global `/arch`, so simply leave the global `/arch` at the
default (SSE2) and raise only the AVX2 TUs.

## Easiest path: use the CMake target

If you build with CMake, just add the repo and link the `marrow` target — the per-file flags above
are already wired up:

```cmake
add_subdirectory(path/to/marrow)
target_link_libraries(your_engine PRIVATE marrow)
```

To consume only the runtime and skip the tests/tools/examples:

```sh
cmake -S marrow -B build -DMRW_BUILD_TESTS=OFF -DMRW_BUILD_TOOLS=OFF -DMRW_BUILD_EXAMPLES=OFF
```

See @ref ref_cmake for all options.

## Other build systems

Compile every file in `src/` plus your code that includes `marrow.h`. Apply the per-ISA flags from
the table above to the six kernel TUs; compile everything else at your baseline. On Unix toolchains
link `-lm` (the math core uses `sqrtf`/`floorf`/`fmodf`). The header compiles as both C11 and C++,
so you can include it from either.

## Binding from another language

The ABI is deliberately flat: `extern "C"`, every output through out-params plus an `mrw_result`
return code, no struct returned by value, fixed-width unsigned counts, and no SIMD types in public
structs. That makes `marrow.h` straightforward to translate into an FFI layer (Rust `bindgen`, Zig
`@cImport`, C# `DllImport`, etc.). Build the runtime as a static or shared library and call it
exactly as C++ would.

## Platforms

Desktop x64 and current/last-gen consoles (little-endian). AVX2 (+FMA) is the primary fast path;
SSE2 is the portable x64 baseline and fallback; scalar is the reference and ultimate fallback.
Determinism is **visual-only** — FMA and reordered reductions are fair game, so results are not
bit-identical across machines (see @ref concept_conventions).
