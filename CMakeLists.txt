cmake_minimum_required(VERSION 3.15)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project(marrow VERSION 0.1.0 LANGUAGES C CXX)

option(MRW_BUILD_TESTS "Build marrow tests" ON)
option(MRW_BUILD_TOOLS "Build marrow offline tools (gltf2marrow, marrow-bake)" ON)
option(MRW_BUILD_DEMO  "Build the Vulkan demo (separate target; needs Vulkan SDK)" OFF)
option(MRW_BUILD_OZZ_BENCH "Build the ozz-animation throughput comparison (FetchContent ozz; never linked into the runtime)" OFF)
# libFuzzer .mrw loader target. Needs Clang/clang-cl; on any other
# compiler it is ignored with a warning (the portable tests/fuzz_replay CTest still gates the corpus).
option(MRW_BUILD_FUZZ "Build the libFuzzer .mrw loader fuzz target (Clang only)" OFF)
# Sanitizer build. Comma list: address,undefined,memory (clang/gcc) or
# address (MSVC). Empty = off. Applied directory-wide below so the whole graph is instrumented.
set(MRW_SANITIZE "" CACHE STRING "Sanitizers to build with (e.g. address,undefined); empty = off")
# Real-asset (glTF) ecological cross-check inside the ozz bench: builds
# ozz's own gltf2ozz importer + marrow's gltf2marrow/mrw_g2m, and compiles the harness's glTF path.
# It needs marrow's offline glTF pieces (the mrw_g2m joint-order DFS + cgltf reader, and the gltf2marrow
# exe for asset prep), which tools/ gates behind MRW_BUILD_TOOLS — so pull both gates in when it is on.
option(MRW_OZZ_BENCH_GLTF "ozz bench: real-asset glTF ecological cross-check (needs ozz tools + gltf2marrow)" OFF)
if (MRW_OZZ_BENCH_GLTF)
  set(MRW_BUILD_OZZ_BENCH ON CACHE BOOL "" FORCE)
  set(MRW_BUILD_TOOLS     ON CACHE BOOL "" FORCE)
endif()

# The fuzz_blob executable lives in the tests subtree, so a fuzz build needs tests on — otherwise
# MRW_BUILD_FUZZ would build only the instrumented marrow_fuzz lib and no exe.
if (MRW_BUILD_FUZZ)
  set(MRW_BUILD_TESTS ON CACHE BOOL "" FORCE)
endif()

# enable_testing() must precede the subdirectories so their add_test() calls register (the tools
# subtree carries the gltf2marrow test).
if (MRW_BUILD_TESTS)
  enable_testing()
endif()

# ---- sanitizers ----
# Directory-wide so the library + tools + tests are all instrumented (placed before the targets and
# add_subdirectory calls so it propagates). NOT compatible with the ISA/banned scans (they assert on
# un-instrumented codegen/symbols) — those CTests are skipped under MRW_SANITIZE (see tests/CMakeLists.txt).
if (MRW_SANITIZE)
  if (MSVC)
    if (NOT MRW_SANITIZE STREQUAL "address")
      message(WARNING "MRW_SANITIZE='${MRW_SANITIZE}': MSVC supports only 'address'; ignoring the rest.")
    endif()
    # /fsanitize=address conflicts with Edit-and-Continue (/ZI from the policy above), the runtime
    # checks (/RTC*) in Debug, and incremental linking — override all three.
    add_compile_options(/fsanitize=address /Zi)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "ProgramDatabase")
    foreach(cfg "" _DEBUG _RELWITHDEBINFO)
      string(REGEX REPLACE "/RTC[1csu]+" "" CMAKE_C_FLAGS${cfg}   "${CMAKE_C_FLAGS${cfg}}")
      string(REGEX REPLACE "/RTC[1csu]+" "" CMAKE_CXX_FLAGS${cfg} "${CMAKE_CXX_FLAGS${cfg}}")
    endforeach()
    add_link_options(/INCREMENTAL:NO)
  else()
    add_compile_options(-fsanitize=${MRW_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fsanitize=${MRW_SANITIZE})
  endif()
  message(STATUS "marrow: building with sanitizers: ${MRW_SANITIZE}")
endif()

# ---- runtime library (pure C11, zero deps) ----
set(MRW_RUNTIME_SOURCES
  src/marrow_math.c
  src/marrow_dispatch.c
  src/marrow_identity.c
  src/marrow_format.c
  src/marrow_pose.c
  src/marrow_blend.c
  src/marrow_ik.c
  src/marrow_batch.c
  src/marrow_batch_sse2.c
  src/marrow_batch_avx2.c
  src/marrow_batch_avx2_fma.c
  src/marrow_blend_batch.c
  src/marrow_blend_sse2.c
  src/marrow_blend_avx2.c
  src/marrow_blend_avx2_fma.c
)
add_library(marrow ${MRW_RUNTIME_SOURCES})
target_include_directories(marrow
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Per-ISA translation units: each kernel TU is compiled with exactly its ISA so the
# baseline can never emit AVX, and the base AVX2 TU never emits FMA. On x64 MSVC, SSE2 is the
# implicit baseline (no /arch needed); /arch:AVX2 enables AVX2, and with default /fp:precise the
# base AVX2 TU emits no FMA because it uses no _mm256_fmadd_* intrinsic. On GCC/Clang we gate
# each TU explicitly and disable contraction so the no-FMA TU stays FMA-free. The SSE2 TU also
# carries -mno-avx/-mno-avx2/-mno-fma so a project-global -mavx/-march=native cannot leak VEX/AVX
# into the baseline (these per-source flags are appended after global ones, so they win). On MSVC
# x64 there is no per-file way to lower below the global /arch; the isa_scan gate is the backstop.
if (MSVC)
  # /arch:AVX2 already enables the F16C intrinsics (vcvtps2ph) used by the f16 palette store.
  set_source_files_properties(src/marrow_batch_avx2.c src/marrow_blend_avx2.c         PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(src/marrow_batch_avx2_fma.c src/marrow_blend_avx2_fma.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # -mf16c lets the AVX2 TUs emit vcvtps2ph for the f16 store; the SSE2/baseline TU keeps -mno-avx
  # (and never compiles the F16C branch), so vcvtps2ph can never appear there (isa_scan enforces it).
  set_source_files_properties(src/marrow_batch_sse2.c src/marrow_blend_sse2.c         PROPERTIES COMPILE_OPTIONS "-msse2;-mno-avx;-mno-avx2;-mno-fma;-mno-f16c")
  set_source_files_properties(src/marrow_batch_avx2.c src/marrow_blend_avx2.c         PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c;-mno-fma;-ffp-contract=off")
  set_source_files_properties(src/marrow_batch_avx2_fma.c src/marrow_blend_avx2_fma.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()

# Per-target µarch knob (src/marrow_batch_simd.h "AVX2 defaults to hardware vgather"): build the AVX2
# kernels with the SSE2-style scalar deinterleave instead of hardware vgather. Same parity, a compile-
# time choice (never a runtime branch). Used to compare hardware-vgather against scalar-deinterleave
# gather strategies at high joint counts; OFF keeps the measured-fastest default.
option(MRW_AVX2_GATHER_SCALAR "Build the AVX2 kernels with scalar deinterleave instead of hardware vgather" OFF)
if (MRW_AVX2_GATHER_SCALAR)
  set_source_files_properties(src/marrow_batch_avx2.c src/marrow_batch_avx2_fma.c
                              src/marrow_blend_avx2.c src/marrow_blend_avx2_fma.c
    PROPERTIES COMPILE_DEFINITIONS MRW_AVX2_GATHER_SCALAR)
endif()

# Non-temporal output stores are deliberately not used: the heavy-rig palette write is raw
# DRAM-write-bandwidth-bound (not read-for-ownership-bound), so streaming stores measure as a non-win
# (≈flat on the heavy rig, a ~29% lean-rig regression). The lever that does cut the write traffic is
# the f16 output palette, which halves the bytes stored.
set_target_properties(marrow PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

# Strict-aliasing is a normative property of the loader. On GCC/Clang we
# build the whole library under -fstrict-aliasing and ask the compiler to flag any
# violation; on MSVC (no equivalent flag) the memcpy-only reads are correct by design.
if (MSVC)
  target_compile_options(marrow PRIVATE /W4)
else()
  target_compile_options(marrow PRIVATE
    -Wall -Wextra -Wpedantic -Wstrict-aliasing=2 -fstrict-aliasing)
endif()

# math.h functions (sqrtf/floorf/fmodf) need libm on Unix toolchains.
if (UNIX)
  target_link_libraries(marrow PUBLIC m)
endif()

# Coverage-instrumented twin of the runtime for the libFuzzer loader target. Built from the SAME
# sources, so the directory-scoped per-ISA set_source_files_properties above
# apply to it too (DRY); adds SanitizerCoverage + ASan/UBSan so libFuzzer gets edge coverage INSIDE
# mrw_blob_open and the validators — instrumenting only the harness TU would leave the fuzzer
# coverage-blind. The normal `marrow` target stays un-instrumented so the ISA-scan and other CI jobs
# see real codegen. Clang-only; the fuzz executable (tests/fuzz_blob.c) links this and adds
# -fsanitize=fuzzer. The portable tests/fuzz_replay gate needs neither and runs everywhere.
if (MRW_BUILD_FUZZ)
  if (CMAKE_C_COMPILER_ID MATCHES "Clang")
    add_library(marrow_fuzz STATIC ${MRW_RUNTIME_SOURCES})
    target_include_directories(marrow_fuzz
      PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_options(marrow_fuzz PRIVATE
      -fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    set_target_properties(marrow_fuzz PROPERTIES
      C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    if (UNIX)
      target_link_libraries(marrow_fuzz PUBLIC m)
    endif()
  else()
    message(WARNING "MRW_BUILD_FUZZ ignored: libFuzzer needs Clang (have "
                    "${CMAKE_C_COMPILER_ID}); the portable fuzz_replay CTest still gates the corpus.")
  endif()
endif()

# Offline tools live in their own targets (heavy libs allowed; never linked into the runtime).
# The shared, checked authoring library is also linked by the tests, so build the tools subtree
# whenever tools, tests, the demo, OR the ozz bench are requested (each links mrw_authoring to
# build procedural .mrw assets in code). tools/ is added before tests/ so its targets exist.
if (MRW_BUILD_TOOLS OR MRW_BUILD_TESTS OR MRW_BUILD_DEMO OR MRW_BUILD_OZZ_BENCH)
  add_subdirectory(tools)
endif()

if (MRW_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Buildable Vulkan demo (separate target; reuses the runtime + authoring/bake, never linked in).
if (MRW_BUILD_DEMO)
  add_subdirectory(demo)
endif()

# ozz-animation throughput comparison (separate gated target; FetchContents ozz, links marrow via
# its C ABI, and is NEVER linked into the runtime — the runtime stays pure-C11 zero-dep).
if (MRW_BUILD_OZZ_BENCH)
  add_subdirectory(bench/ozz_compare)
endif()
