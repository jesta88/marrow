# marrow vs ozz-animation throughput comparison - a SEPARATE gated target (like tools/ and demo/).
# FetchContents ozz-animation (Apache-2.0) and links it next to the marrow runtime via marrow's C
# ABI. The marrow runtime (src/, marrow.h) is NEVER touched and NEVER links ozz - this target only
# *consumes* the already-built marrow library.

include(FetchContent)

# --- ozz-animation: build ONLY the libraries we need (no tools/samples/tests/fbx). ---
#  ozz_base               - platform, memory, maths, io
#  ozz_animation          - runtime SamplingJob / LocalToModelJob (the timed path)
#  ozz_animation_offline  - RawSkeleton/RawAnimation + builders + AnimationOptimizer (asset setup only)
# ozz_geometry is NOT needed: skinning is our own shared routine.
# The real-asset cross-check (MRW_OZZ_BENCH_GLTF) needs ozz's OWN offline glTF importer: gltf2ozz
# (+ ozz_animation_tools, which pulls jsoncpp + tinygltf). Built ONLY for that cross-check; the
# default procedural bench keeps the lean OFF set so its build is unchanged. The harness still links
# only ozz_animation/ozz_base at runtime (it loads the .ozz archives gltf2ozz produces).
if (MRW_OZZ_BENCH_GLTF)
  set(ozz_build_tools  ON  CACHE BOOL "" FORCE)
  set(ozz_build_gltf   ON  CACHE BOOL "" FORCE)
else()
  set(ozz_build_tools  OFF CACHE BOOL "" FORCE)
  set(ozz_build_gltf   OFF CACHE BOOL "" FORCE)
endif()
set(ozz_build_fbx      OFF CACHE BOOL "" FORCE)
set(ozz_build_samples  OFF CACHE BOOL "" FORCE)
set(ozz_build_howtos   OFF CACHE BOOL "" FORCE)
set(ozz_build_tests    OFF CACHE BOOL "" FORCE)
set(ozz_build_simd_ref OFF CACHE BOOL "" FORCE)   # we want ozz's REAL SSE/AVX SIMD, not the scalar ref
# Match the rest of the project's MSVC runtime (CMake default = dynamic /MD). ozz defaults to /MT,
# which would clash with marrow at link time; force the DLL runtime so both agree.
set(ozz_build_msvc_rt_dll ON CACHE BOOL "" FORCE)

FetchContent_Declare(ozz-animation
  GIT_REPOSITORY https://github.com/guillaumeblanc/ozz-animation.git
  GIT_TAG        0.16.0
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(ozz-animation)

# ------------------------------------------------------------------ harness executable
add_executable(marrow_ozz_bench
  main.cpp
  # Reuse the demo's ONE monotonic timer + the public prof_stat percentile machinery, and the demo
  # thread pool - compiled straight in (both are pure C, no Vulkan). Their C headers are included
  # under extern "C" by the harness.
  ${CMAKE_SOURCE_DIR}/demo/profiler.c
  ${CMAKE_SOURCE_DIR}/demo/jobs.c
)

target_include_directories(marrow_ozz_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/demo)        # profiler.h / jobs.h

# marrow via its C ABI; mrw_authoring to build the procedural .mrw in code; ozz runtime + offline.
target_link_libraries(marrow_ozz_bench PRIVATE
  marrow mrw_authoring
  ozz_base ozz_animation ozz_animation_offline)

set_target_properties(marrow_ozz_bench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# AVX2(+FMA) for the HARNESS + ozz only - NEVER project-global: a global /arch
# would let marrow's SSE2/scalar TUs emit AVX and break the ISA/disassembly gate. marrow is
# linked as already built. ozz's own targets pick up the same /arch so its math uses its best path.
if (MSVC)
  target_compile_options(marrow_ozz_bench PRIVATE /arch:AVX2 /W3)
  foreach(t ozz_base ozz_animation ozz_animation_offline)
    if (TARGET ${t})
      target_compile_options(${t} PRIVATE /arch:AVX2)
    endif()
  endforeach()
else()
  target_compile_options(marrow_ozz_bench PRIVATE -mavx2 -mfma -Wall -Wextra)
endif()

if (MRW_BUILD_TESTS)
  # Tiny smoke run so the bench can't rot (gated like the rest). The full sweep is run by hand.
  add_test(NAME ozz_bench_smoke COMMAND marrow_ozz_bench --smoke)
endif()

# ------------------------------------------------------------------ real-asset glTF cross-check
# When MRW_OZZ_BENCH_GLTF is ON the harness compiles its --gltf path (real-asset glTF cross-check):
#  - link mrw_g2m for the shared skin→marrow joint-order DFS (mrw_g2m_joint_order_build) and the
#    SINGLE vendored cgltf translation unit (mrw_cgltf, PUBLIC-linked by mrw_g2m) used to read the
#    skin inverse-bind matrices - never a second cgltf in the link. mrw_g2m lives in tools/, which the
#    top-level forces ON together with this option, and tools/ is added before this subdirectory.
#  - depend on both offline importers so a bench build also builds the asset-prep tools: ozz's own
#    gltf2ozz (built because we flipped ozz_build_tools/gltf ON above) and marrow's gltf2marrow.
# OFF ⇒ the harness's glTF code is #ifdef'd out; nothing here touches the frozen procedural bench.
if (MRW_OZZ_BENCH_GLTF)
  target_compile_definitions(marrow_ozz_bench PRIVATE MRW_OZZ_BENCH_GLTF=1)
  target_link_libraries(marrow_ozz_bench PRIVATE mrw_g2m)
  add_dependencies(marrow_ozz_bench gltf2ozz gltf2marrow)
  # "compute-only" floor: a second copy of the AVX2+FMA kernel
  # built from the shared src/ source with MRW_BENCH_NO_SCATTER (palette write removed). Bench-only -
  # the marrow runtime never sees this TU or that define. Needs src/ for marrow_batch_simd.h + internals.
  target_sources(marrow_ozz_bench PRIVATE floor_noscatter.c)
  target_include_directories(marrow_ozz_bench PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # The shared kernel source uses C11 (_Static_assert / _Alignas); MSVC's default C dialect rejects
  # them, so build this TU as C11 like the runtime library does (GCC/Clang default already allows it).
  if (MSVC)
    set_source_files_properties(floor_noscatter.c PROPERTIES COMPILE_OPTIONS "/std:c11")
  endif()
endif()
