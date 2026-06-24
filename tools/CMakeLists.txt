# Offline tool targets. These are SEPARATE from the zero-dependency `marrow` runtime: they may use
# heavy libraries (cgltf) and reuse the runtime's scalar math + identity hashing, but the runtime
# never links them. The banned-construct / ISA scans stay scoped to src/, so nothing here can
# regress the runtime's guarantees.

# ---- shared, checked .mrw authoring library (lifted from tests/) ----
# Both offline tools (gltf2marrow, marrow-bake) and the tests serialize blobs through this. It is
# the checked counterpart of the old test helper: a mrw_result API with 64-bit overflow checks and
# OOM-safe allocation, with the byte-emitting core preserved (golden-bytes parity).
add_library(mrw_authoring STATIC
  authoring/mrw_authoring.c
  authoring/mrw_decompose.c
)
target_include_directories(mrw_authoring
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/authoring
  PRIVATE ${CMAKE_SOURCE_DIR}/src)          # marrow_internal.h (wire constants, not public ABI)
target_link_libraries(mrw_authoring PUBLIC marrow)
set_target_properties(mrw_authoring PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
if (MSVC)
  target_compile_options(mrw_authoring PRIVATE /W4)
else()
  target_compile_options(mrw_authoring PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- Tier-B bake core (lifted from the tests; always built, like mrw_authoring) ----
# Allocation-free (caller-owned scratch + texels): the front-end and the runtime parity gate both
# link it. It uses a local f32->half and references only mrw_decompose from the authoring lib, so its
# symbol closure pulls in no allocator object. No cgltf dependency - that lives in the front-end.
add_library(mrw_bake STATIC bake/mrw_bake.c)
target_include_directories(mrw_bake
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/bake
  PRIVATE ${CMAKE_SOURCE_DIR}/src)            # marrow_internal.h (§3.4/§3.5 helpers)
target_link_libraries(mrw_bake PUBLIC marrow mrw_authoring)
set_target_properties(mrw_bake PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
if (MSVC)
  target_compile_options(mrw_bake PRIVATE /W4)
else()
  target_compile_options(mrw_bake PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- tool executables + glTF libs (the authoring/bake libs above are always available to the
# tests) ----
# Built for the offline tools AND for the Vulkan demo: the demo links mrw_cgltf + mrw_g2m (for the
# shared joint-order DFS and the cgltf mesh reader) and runs gltf2marrow + marrow-bake as its
# asset-prep step. So gate on MRW_BUILD_DEMO too, not just MRW_BUILD_TOOLS.
if (MRW_BUILD_TOOLS OR MRW_BUILD_DEMO)
  # Vendored cgltf, compiled as a single implementation TU at a lowered warning level (we do not
  # patch vendored code). PUBLIC include exposes cgltf.h to every tool that links it.
  add_library(mrw_cgltf STATIC third_party/cgltf_impl.c)
  target_include_directories(mrw_cgltf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
  set_target_properties(mrw_cgltf PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  if (MSVC)
    target_compile_options(mrw_cgltf PRIVATE /W0)
  else()
    target_compile_options(mrw_cgltf PRIVATE -w)
  endif()
  if (UNIX)
    target_link_libraries(mrw_cgltf PUBLIC m)   # cgltf uses libm (powf/atof internals)
  endif()

  # Conversion core as a library so the CLI, the test, AND the demo link it. joint_order.c is the
  # shared skin-order → marrow-index DFS, called by both the converter and the demo mesh loader.
  add_library(mrw_g2m STATIC gltf2marrow/convert.c gltf2marrow/joint_order.c)
  target_include_directories(mrw_g2m PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gltf2marrow)
  target_link_libraries(mrw_g2m PUBLIC marrow mrw_authoring mrw_cgltf)
  set_target_properties(mrw_g2m PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  if (MSVC)
    target_compile_options(mrw_g2m PRIVATE /W4)
  else()
    target_compile_options(mrw_g2m PRIVATE -Wall -Wextra)
  endif()

  add_executable(gltf2marrow gltf2marrow/main.c)
  target_link_libraries(gltf2marrow PRIVATE mrw_g2m)
  set_target_properties(gltf2marrow PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  if (MSVC)
    target_compile_options(gltf2marrow PRIVATE /W4)
  else()
    target_compile_options(gltf2marrow PRIVATE -Wall -Wextra)
  endif()

  if (MRW_BUILD_TESTS)
    add_executable(test_gltf2marrow gltf2marrow/tests/test_gltf2marrow.c)
    target_link_libraries(test_gltf2marrow PRIVATE mrw_g2m)
    target_compile_definitions(test_gltf2marrow PRIVATE MRW_G2M_TESTDIR="${CMAKE_CURRENT_BINARY_DIR}")
    set_target_properties(test_gltf2marrow PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    add_test(NAME test_gltf2marrow COMMAND test_gltf2marrow)
  endif()

  # marrow-bake front-end: I/O + struct reconstruction + probe geometry + assembly (bake_run.c) and
  # the --mesh probe path (mesh_probes.c, cgltf). The allocation-free bake CORE (mrw_bake) is built
  # outside this gate and linked here; cgltf only enters through this front-end, never the runtime.
  add_library(mrw_baketool STATIC bake/bake_run.c bake/mesh_probes.c)
  target_include_directories(mrw_baketool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/bake)
  target_link_libraries(mrw_baketool PUBLIC marrow mrw_authoring mrw_bake mrw_cgltf)
  set_target_properties(mrw_baketool PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  if (MSVC)
    target_compile_options(mrw_baketool PRIVATE /W4)
  else()
    target_compile_options(mrw_baketool PRIVATE -Wall -Wextra)
  endif()

  add_executable(marrow-bake bake/main.c)
  target_link_libraries(marrow-bake PRIVATE mrw_baketool)
  set_target_properties(marrow-bake PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  if (MSVC)
    target_compile_options(marrow-bake PRIVATE /W4)
  else()
    target_compile_options(marrow-bake PRIVATE -Wall -Wextra)
  endif()

  if (MRW_BUILD_TESTS)
    add_executable(test_bake bake/tests/test_bake.c)
    target_link_libraries(test_bake PRIVATE mrw_baketool)
    target_compile_definitions(test_bake PRIVATE MRW_BAKE_TESTDIR="${CMAKE_CURRENT_BINARY_DIR}")
    set_target_properties(test_bake PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    add_test(NAME test_bake COMMAND test_bake)
  endif()
endif()
