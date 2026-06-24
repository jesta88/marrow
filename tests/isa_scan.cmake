# ISA/disassembly scan: prove the per-ISA translation units stay in their lane -
#   - the SSE2/baseline TU contains NO AVX - neither ymm/zmm NOR any VEX-encoded 128-bit AVX op
#     (a global -mavx/-march=native could otherwise emit `vmulps xmm…`, which a bare `ymm` grep
#     would miss while still breaking the "baseline never emits AVX" contract),
#   - the AVX2-without-FMA TU contains NO FMA (no vfmadd),
#   - the AVX2+FMA TU DOES use FMA (vfmadd) - i.e. the split actually buys fused multiply-add.
# Best-effort by default: needs dumpbin (MSVC) or objdump (GNU); a missing disassembler / object / empty
# disasm is a SKIP so the gate never produces a false negative on an unsupported toolchain. CI passes
# -DMRW_SCAN_STRICT=ON: there a skip becomes a FAILURE, so a job that claims to run the ISA scan can't
# go green by silently skipping it.
#
# Invoked: cmake -DMRW_OBJDIR=<marrow.dir/src> [-DMRW_SCAN_STRICT=ON] -P isa_scan.cmake

find_program(DUMPBIN dumpbin)
find_program(OBJDUMP objdump)

set(errs "")

# A skip is fatal under strict, an informational note otherwise. (macro ⇒ edits `errs` in this scope.)
macro(skip_or_fail msg)
  if(MRW_SCAN_STRICT)
    list(APPEND errs "${msg}")
  else()
    message(STATUS "isa_scan: ${msg} - skipped")
  endif()
endmacro()

if(NOT DUMPBIN AND NOT OBJDUMP)
  if(MRW_SCAN_STRICT)
    message(FATAL_ERROR "isa_scan (strict): no disassembler (dumpbin/objdump) found")
  endif()
  message(STATUS "isa_scan: no disassembler (dumpbin/objdump) found - skipped")
  return()
endif()

function(disasm obj out_var)
  set(text "")
  if(DUMPBIN)
    execute_process(COMMAND "${DUMPBIN}" /disasm:nobytes "${obj}" OUTPUT_VARIABLE text ERROR_QUIET)
  else()
    execute_process(COMMAND "${OBJDUMP}" -d "${obj}" OUTPUT_VARIABLE text ERROR_QUIET)
  endif()
  set(${out_var} "${text}" PARENT_SCOPE)
endfunction()

function(find_obj stem out_var)
  file(GLOB hits "${MRW_OBJDIR}/${stem}.c.obj" "${MRW_OBJDIR}/${stem}.c.o")
  if(hits)
    list(GET hits 0 first)
    set(${out_var} "${first}" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()

# Any of these mnemonics in a "baseline" object means AVX leaked in: 256/512-bit registers, or a
# VEX-encoded 128-bit AVX op (the legacy SSE forms have no `v` prefix, so this never matches SSE2).
set(AVX_MARKERS "ymm|zmm|vmovaps|vmovups|vmulps|vaddps|vsubps|vdivps|vsqrtps|vandps|vandnps|vorps|vxorps|vcmpps|vblendvps|vbroadcast|vfmadd|vcvtps2ph")

# The clip-batch (marrow_batch_*) and the pose-combine (marrow_blend_*) SIMD TUs share the same
# per-ISA contract and the same `wide` source, so both stems are scanned with identical rules.
foreach(stem "marrow_batch_sse2" "marrow_blend_sse2")
  find_obj("${stem}" obj)
  if(obj)
    disasm("${obj}" d)
    if(d STREQUAL "")
      skip_or_fail("${stem} disasm empty")
    elseif(d MATCHES "${AVX_MARKERS}")
      list(APPEND errs "SSE2/baseline TU emits AVX (ymm/zmm or a VEX-encoded op): ${obj}")
    endif()
  else()
    skip_or_fail("${stem} object not found under ${MRW_OBJDIR}")
  endif()
endforeach()

foreach(stem "marrow_batch_avx2" "marrow_blend_avx2")   # exact stem; does NOT match *_avx2_fma
  find_obj("${stem}" obj)
  if(obj)
    disasm("${obj}" d)
    if(d STREQUAL "")
      skip_or_fail("${stem} disasm empty")
    elseif(d MATCHES "vfmadd")
      list(APPEND errs "AVX2-without-FMA TU emits FMA (vfmadd): ${obj}")
    endif()
  else()
    skip_or_fail("${stem} object not found under ${MRW_OBJDIR}")
  endif()
endforeach()

foreach(stem "marrow_batch_avx2_fma" "marrow_blend_avx2_fma")
  find_obj("${stem}" obj)
  if(obj)
    disasm("${obj}" d)
    if(d STREQUAL "")
      skip_or_fail("${stem} disasm empty")
    elseif(NOT d MATCHES "vfmadd")
      list(APPEND errs "AVX2+FMA TU has no vfmadd - the FMA split buys nothing: ${obj}")
    endif()
  else()
    skip_or_fail("${stem} object not found under ${MRW_OBJDIR}")
  endif()
endforeach()

if(errs)
  string(REPLACE ";" "\n  " pretty "${errs}")
  message(FATAL_ERROR "isa_scan failed:\n  ${pretty}")
endif()
message(STATUS "isa_scan: clean")
