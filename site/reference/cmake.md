CMake options {#ref_cmake}
=============

marrow builds with CMake (Ninja recommended). The runtime itself has no third-party dependencies.

```sh
cmake -S . -B out/build -G Ninja
cmake --build out/build
ctest --test-dir out/build --output-on-failure
```

On Windows with Visual Studio, the supplied `CMakePresets.json` wraps this (`cmake --preset
x64-release`, etc.).

## Options

| Option | Default | Builds / does |
|---|---|---|
| `MRW_BUILD_TESTS` | `ON` | The CTest suite (parity, golden bytes, malformed-input, fuzz replay, the ISA/banned scans). |
| `MRW_BUILD_TOOLS` | `ON` | The offline tools `gltf2marrow` and `marrow-bake`. |
| `MRW_BUILD_EXAMPLES` | `ON` | The runnable API examples under `examples/` (e.g. `mrw_quickstart`). |
| `MRW_BUILD_DEMO` | `OFF` | The GLFW + Vulkan demo (needs the Vulkan SDK). |
| `MRW_BUILD_OZZ_BENCH` | `OFF` | The ozz-animation throughput comparison (FetchContents ozz; never linked into the runtime). |
| `MRW_BUILD_FUZZ` | `OFF` | The libFuzzer `.mrw` loader fuzz target (Clang / clang-cl only). |
| `MRW_SANITIZE` | *(off)* | Sanitizer instrumentation, comma-separated, applied directory-wide — e.g. `-DMRW_SANITIZE=address,undefined` (`address` only on MSVC). |

## Consuming only the runtime

To build just the library for embedding, turn the rest off:

```sh
cmake -S . -B build -DMRW_BUILD_TESTS=OFF -DMRW_BUILD_TOOLS=OFF -DMRW_BUILD_EXAMPLES=OFF
```

Then link the `marrow` target, or compile `src/*.c` into your own build. The per-translation-unit
ISA compile flags are the one thing to get right when you don't use the CMake target — see
@ref howto_integrate.
