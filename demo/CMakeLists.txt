# marrow Vulkan demo - a SEPARATE target (like tools/): may use heavy deps (Vulkan, GLFW, volk,
# cgltf) and is never linked into the zero-dependency runtime. Gated by MRW_BUILD_DEMO in the root.

find_package(Vulkan REQUIRED)

include(FetchContent)

# --- GLFW (window/surface/input) ---
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
  GIT_SHALLOW    TRUE)

# --- volk (Vulkan meta-loader: loads core 1.4 + all extension entrypoints, incl. shader objects) ---
FetchContent_Declare(volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG        vulkan-sdk-1.4.341.0
  GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(glfw volk)

# ------------------------------------------------------------------ shader compilation + embed
# glslc compiles each GLSL stage to SPIR-V (Vulkan 1.3 / SPIR-V 1.6), then embed_spirv.cmake turns
# the .spv into a uint32 C array header so the executable is self-contained (no runtime file IO).
set(SHADER_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_GEN_DIR})

set(DEMO_SHADER_HEADERS "")
# Extra args after NAME (e.g. -DMARROW_BENCH) are passed straight to glslc, so one source can be
# compiled into several SPIR-V variants.
function(add_demo_shader SRC STAGE NAME)
  set(spv ${SHADER_GEN_DIR}/${NAME}.spv)
  set(hdr ${SHADER_GEN_DIR}/${NAME}.spv.h)
  add_custom_command(
    OUTPUT  ${spv}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} -fshader-stage=${STAGE} --target-env=vulkan1.3 -O ${ARGN}
            ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} -o ${spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}
    COMMENT "glslc ${SRC} ${ARGN}"
    VERBATIM)
  add_custom_command(
    OUTPUT  ${hdr}
    COMMAND ${CMAKE_COMMAND} -DSPV=${spv} -DHDR=${hdr} -DSYM=${NAME}_spv
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
    DEPENDS ${spv} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
    COMMENT "embed ${NAME}.spv"
    VERBATIM)
  set(DEMO_SHADER_HEADERS ${DEMO_SHADER_HEADERS} ${hdr} PARENT_SCOPE)
endfunction()

add_demo_shader(shaders/ground.vert vert ground_vert)
add_demo_shader(shaders/ground.frag frag ground_frag)
add_demo_shader(shaders/crowd.vert     vert crowd_vert)
add_demo_shader(shaders/crowd.frag     frag crowd_frag)
# Bench-only vertex-skinning microbenchmark variants (rasterizer-discard differential timing):
add_demo_shader(shaders/crowd.vert            vert crowd_skin_bench_vert -DMARROW_BENCH)
add_demo_shader(shaders/crowd_static_bench.vert vert crowd_static_bench_vert)
add_demo_shader(shaders/skin_tierA.vert vert skin_tierA_vert)
add_demo_shader(shaders/skin_tierA_f16.vert vert skin_tierA_f16_vert)
# Bone-line skeleton render LOD, drawn as a LINE_LIST: the far tail poses from the Tier-B baked
# palette (crowd_skel.vert), the near set from the CPU Tier-A palette SSBO (skin_tierA_skel.vert).
add_demo_shader(shaders/crowd_skel.vert      vert crowd_skel_vert)
add_demo_shader(shaders/skin_tierA_skel.vert vert skin_tierA_skel_vert)
add_demo_shader(shaders/skel.frag       frag skel_frag)
add_demo_shader(shaders/validate_skin.comp comp validate_skin_comp)
add_demo_shader(shaders/validate_skin_f16.comp comp validate_skin_f16_comp)
add_demo_shader(shaders/validate_skin_f16_ssbo.comp comp validate_skin_f16_ssbo_comp)
add_demo_shader(shaders/hud.vert       vert hud_vert)
add_demo_shader(shaders/hud.frag       frag hud_frag)

add_custom_target(demo_shaders DEPENDS ${DEMO_SHADER_HEADERS})

# ------------------------------------------------------------------ executable
add_executable(marrow_demo
  main.c
  vk_context.c
  camera.c
  assets_proc.c
  assets_gltf.c
  models.c
  crowd.c
  crowd_cpu.c
  heroes.c
  field.c
  jobs.c
  validate.c
  profiler.c
  hud.c
)
add_dependencies(marrow_demo demo_shaders)

target_include_directories(marrow_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SHADER_GEN_DIR})

# volk owns the Vulkan entrypoints (we link it, NOT Vulkan::Vulkan, to avoid double prototypes).
# mrw_g2m + mrw_cgltf bring the shared joint-order DFS (joint_order.c) and the cgltf glTF reader used
# by assets_gltf.c; their tool libs are made available for the demo by the MRW_BUILD_DEMO gate in
# tools/CMakeLists.txt. Linking mrw_g2m pulls only the objects assets_gltf.c references (joint_order),
# not the converter front-end.
target_link_libraries(marrow_demo PRIVATE marrow mrw_authoring mrw_bake mrw_g2m mrw_cgltf glfw volk)
# Thread pool (demo/jobs.c) - the demo schedules marrow's batch across cores; the runtime stays
# thread-free. Win32 threads are in the platform libs; POSIX needs pthread (Threads::Threads).
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(marrow_demo PRIVATE Threads::Threads)
if (UNIX)
  target_link_libraries(marrow_demo PRIVATE ${CMAKE_DL_LIBS})
endif()

set_target_properties(marrow_demo PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
if (MSVC)
  target_compile_options(marrow_demo PRIVATE /W3)
else()
  target_compile_options(marrow_demo PRIVATE -Wall -Wextra)
endif()

# ------------------------------------------------------------------ tests (gated like the rest)
# enable_testing() is called in the root when MRW_BUILD_TESTS is ON, so these register only then.
if (MRW_BUILD_TESTS)
  # CPU-only gate: parity checks + scalar/SSE2/AVX2 SIMD parity + CPU-crowd palette layout parity.
  # No window, no GPU, no network - safe everywhere the demo target builds.
  add_test(NAME demo_validate COMMAND marrow_demo --validate)

  # Headless --bench --smoke: exercises the bench path, GPU-timestamp readback, and both crowd
  # tiers, then re-parses the emitted CSV/JSON (schema gate). Needs a Vulkan device + a window-
  # system surface, exactly like --validate-gpu.
  add_test(NAME demo_bench_smoke COMMAND ${CMAKE_COMMAND}
    -DEXE=$<TARGET_FILE:marrow_demo>
    -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_bench.cmake)
  # Needs a Vulkan device + WSI surface; label it so headless runners can `ctest -LE gpu`.
  set_tests_properties(demo_bench_smoke PROPERTIES LABELS gpu)

  # Headless field+skeleton smoke: drive the unified LOD field for a few fixed-step frames in both
  # render modes (near Tier-A mesh + far Tier-B mesh band + far bone-line tail, then global all-
  # skeleton) and assert each band populated, so a regression that drops a draw fails. Needs a
  # Vulkan device + WSI surface like the bench smoke.
  add_test(NAME demo_field_smoke COMMAND marrow_demo --field-smoke)
  set_tests_properties(demo_field_smoke PROPERTIES LABELS gpu)
endif()

# ------------------------------------------------------------------ glTF asset prep (UAL1)
# Opt-in target `demo_ual1`: run the committed UAL1 game character (assets/UAL1_Standard_RM.glb - a
# 65-bone Quaternius rig, CC0) through the real offline pipeline (gltf2marrow -> marrow-bake --mesh)
# to produce the .mrw the demo loads with
#   marrow_demo --gltf <build>/demo/assets/UAL1_Standard_RM.glb
# (the demo derives the sibling .mrw automatically). The .glb is self-contained (no external .bin) and
# committed in-tree, so unlike the old CesiumMan target this needs no network. It is copied into the
# build assets dir so the produced rig/baked .mrw sit beside it (the demo's sibling-derivation) and the
# source tree stays clean. Of UAL1's 43 animations we bake two locomotion loops in role order
# (Walk_Loop -> clip 0 "walk", Sprint_Loop -> clip 1 "run") - what the heroes cross-fade and the crowd cycles.
set(UAL1_SRC_GLB ${CMAKE_SOURCE_DIR}/assets/UAL1_Standard_RM.glb)
set(UAL1_DIR     ${CMAKE_CURRENT_BINARY_DIR}/assets)
set(UAL1_GLB     ${UAL1_DIR}/UAL1_Standard_RM.glb)
set(UAL1_RIG     ${UAL1_DIR}/UAL1_Standard_RM_rig.mrw)
set(UAL1_MRW     ${UAL1_DIR}/UAL1_Standard_RM.mrw)

add_custom_command(
  OUTPUT  ${UAL1_GLB}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${UAL1_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${UAL1_SRC_GLB} ${UAL1_GLB}
  DEPENDS ${UAL1_SRC_GLB}
  COMMENT "Copying UAL1_Standard_RM.glb into ${UAL1_DIR}"
  VERBATIM)

add_custom_command(
  OUTPUT  ${UAL1_RIG}
  COMMAND gltf2marrow ${UAL1_GLB} -o ${UAL1_RIG} --loop --anim Walk_Loop --anim Sprint_Loop
  DEPENDS gltf2marrow ${UAL1_GLB}
  COMMENT "gltf2marrow: UAL1 -> rig .mrw (skeleton + Walk_Loop/Sprint_Loop clips)"
  VERBATIM)

add_custom_command(
  OUTPUT  ${UAL1_MRW}
  COMMAND marrow-bake ${UAL1_RIG} -o ${UAL1_MRW} --mesh ${UAL1_GLB}
  DEPENDS marrow-bake ${UAL1_RIG} ${UAL1_GLB}
  COMMENT "marrow-bake: UAL1 rig -> baked Tier-B .mrw"
  VERBATIM)

add_custom_target(demo_ual1 DEPENDS ${UAL1_MRW})

# Bake UAL1 as part of the demo build so the runtime model switch (M) always has at least one model
# beyond the procedural biped to cycle to. The source .glb is committed (no network) and the bake is
# a no-op once up to date, so this is cheap; the demo scans this build's assets dir at startup. Drop
# any other gltf2marrow->marrow-bake output into that folder and it joins the cycle too.
add_dependencies(marrow_demo demo_ual1)
