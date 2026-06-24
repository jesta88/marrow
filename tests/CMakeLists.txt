# Test glue: thin trusted-input wrappers over the lifted authoring lib (mrw_build.c). The checked
# serializer and the matrix↔TRS decompose math now live in tools/authoring; the Tier-B bake core
# lives in tools/bake (mrw_bake, linked below) now that marrow-bake has landed. These files keep the
# legacy test API so existing call sites and golden fixtures are unchanged.
add_library(mrw_build STATIC mrw_build.c)
target_include_directories(mrw_build
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mrw_build PUBLIC marrow mrw_authoring mrw_bake)
set_target_properties(mrw_build PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)

function(mrw_test name)
  add_executable(${name} ${name}.c)
  target_link_libraries(${name} PRIVATE marrow mrw_build)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
  set_target_properties(${name} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

mrw_test(test_math)
mrw_test(test_format_pose)
mrw_test(test_sampling)
mrw_test(test_root_motion)
mrw_test(test_identity)
mrw_test(test_malformed)
mrw_test(test_batch)
mrw_test(test_parity)
mrw_test(test_blend)
mrw_test(test_additive)
mrw_test(test_ik)
mrw_test(test_blend_batch)

# f16 encoder bit-identity gate vs the F16C vcvtps2ph instruction. Compiled with F16C so the
# intrinsic is available; the test self-skips at runtime when the host lacks F16C.
if (MSVC)
  set_source_files_properties(test_f16c.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  set_source_files_properties(test_f16c.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c")
endif()
mrw_test(test_f16c)

# f16 end-to-end skinning precision gate: decode the f16 palette,
# skin a synthetic character-scale mesh, assert vertex/normal error within the owner-ratified bounds.
mrw_test(test_f16_skin)

# codec-1 (scale-free) gates: round-trip, id-equivalence, bake parity, malformed truncation.
# Needs the bake-core + decompose headers (the bake-parity case links mrw_bake via mrw_build).
mrw_test(test_codec1)
target_include_directories(test_codec1 PRIVATE
  ${CMAKE_SOURCE_DIR}/tools/bake ${CMAKE_SOURCE_DIR}/tools/authoring)
mrw_test(test_golden_bytes)
target_compile_definitions(test_golden_bytes PRIVATE MRW_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/fixtures")

# Batch throughput benchmark - built always; CTest runs only the tiny --smoke config so it
# can't bit-rot. Run the executable directly (no args) for the full bone-count/batch-size table.
add_executable(bench_batch bench_batch.c)
target_link_libraries(bench_batch PRIVATE marrow mrw_build)
target_include_directories(bench_batch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
set_target_properties(bench_batch PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
add_test(NAME bench_batch_smoke COMMAND bench_batch --smoke)

# C++ compile-and-link of the public header (verifies extern "C" + C++ compatibility).
add_executable(test_cpp_link test_cpp_link.cpp)
target_link_libraries(test_cpp_link PRIVATE marrow)
target_include_directories(test_cpp_link PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(test_cpp_link PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
add_test(NAME test_cpp_link COMMAND test_cpp_link)

# Fixture generator (writes committed fixtures/ files when run with --emit).
add_executable(gen_fixtures gen_fixtures.c)
target_link_libraries(gen_fixtures PRIVATE marrow mrw_build)
target_include_directories(gen_fixtures PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
set_target_properties(gen_fixtures PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)

# ---- .mrw loader fuzzing ----
# Seed-corpus generator: writes tests/fuzz/corpus/*.mrw (one valid blob per loader shape). Run once
# and commit the output; the fuzz_replay gate then keeps the corpus honest.
add_executable(gen_fuzz_corpus gen_fuzz_corpus.c)
target_link_libraries(gen_fuzz_corpus PRIVATE marrow mrw_build)
target_include_directories(gen_fuzz_corpus PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
set_target_properties(gen_fuzz_corpus PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)

# Portable corpus replayer + permanent regression gate: builds on ANY toolchain (no libFuzzer), runs
# the shared mrw_fuzz_one over every committed corpus + crash file, and asserts corpus coverage. Picks
# up ASan/UBSan automatically under MRW_SANITIZE.
add_executable(fuzz_replay fuzz_replay.c fuzz_blob.c)
target_link_libraries(fuzz_replay PRIVATE marrow)
target_include_directories(fuzz_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
set_target_properties(fuzz_replay PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
file(GLOB MRW_FUZZ_CORPUS  CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/corpus/*.mrw)
file(GLOB MRW_FUZZ_CRASHES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/crash-*)
add_test(NAME fuzz_replay COMMAND fuzz_replay --check-coverage ${MRW_FUZZ_CORPUS} ${MRW_FUZZ_CRASHES})

# libFuzzer target - only when MRW_BUILD_FUZZ + Clang produced the instrumented marrow_fuzz twin.
# Not a CTest (it runs unbounded); CI runs it with -max_total_time. Local/MSVC coverage is fuzz_replay.
if (TARGET marrow_fuzz)
  add_executable(fuzz_blob fuzz_blob.c)
  target_link_libraries(fuzz_blob PRIVATE marrow_fuzz)
  target_include_directories(fuzz_blob PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
  # -fno-sanitize-recover=all so a UBSan finding ABORTS (clang UBSan is recoverable by default - the
  # fuzzer would otherwise print a runtime-error report and still exit 0 after -max_total_time).
  target_compile_options(fuzz_blob PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(fuzz_blob    PRIVATE -fsanitize=fuzzer,address,undefined)
  set_target_properties(fuzz_blob PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
endif()

# Env-gated CI honesty check: proves AVX2/FMA was actually
# executed on a job that claims to test SIMD (parity tests silently skip absent backends). With
# MRW_REQUIRE_AVX2 unset it is a host-independent pass, so it never breaks a non-AVX2 host / the Jaguar
# baseline locally - only the CI SIMD job sets the env and enforces it.
mrw_test(test_require_avx2)

# CI passes -DMRW_SCAN_STRICT=ON so a missing disassembler / symbol tool / object FAILS these scans
# rather than silently skipping; locally it defaults OFF (best-effort).
option(MRW_SCAN_STRICT "CI: fail (don't skip) the ISA/banned binary scans when tooling/objects are missing" OFF)

# The ISA/banned/bake scans assert on UN-instrumented codegen + symbols (the AVX/FMA disassembly, and
# the absence of malloc/free) - sanitizers add interceptors and change codegen, so these are skipped
# under MRW_SANITIZE (a clean CI job still runs them).
if (NOT MRW_SANITIZE)
  # Banned-construct + symbol scan over the runtime sources/object.
  add_test(NAME banned_scan
    COMMAND ${CMAKE_COMMAND}
      -DMRW_SRC_DIR=${CMAKE_SOURCE_DIR}/src
      -DMRW_LIB=$<TARGET_FILE:marrow>
      -DMRW_SCAN_STRICT=${MRW_SCAN_STRICT}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/banned_scan.cmake)

  # The bake CORE is allocation-free (caller-owned scratch + texels): scan just that TU (the front-end
  # bake_run.c / main.c legitimately allocate, so they are excluded). Both the source and the object.
  add_test(NAME bake_core_scan
    COMMAND ${CMAKE_COMMAND}
      "-DMRW_SRC_FILES=${CMAKE_SOURCE_DIR}/tools/bake/mrw_bake.c;${CMAKE_SOURCE_DIR}/tools/bake/mrw_bake.h"
      -DMRW_LIB=$<TARGET_FILE:mrw_bake>
      -DMRW_SCAN_STRICT=${MRW_SCAN_STRICT}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/banned_scan.cmake)

  # ISA/disassembly scan: SSE2 TU emits no AVX; AVX2-without-FMA TU emits no FMA; the
  # AVX2+FMA TU does use FMA. Best-effort locally; strict (no silent skip) under -DMRW_SCAN_STRICT=ON.
  add_test(NAME isa_scan
    COMMAND ${CMAKE_COMMAND}
      -DMRW_OBJDIR=${CMAKE_BINARY_DIR}/CMakeFiles/marrow.dir/src
      -DMRW_SCAN_STRICT=${MRW_SCAN_STRICT}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/isa_scan.cmake)
endif()
