# Banned-construct + symbol scan for the marrow runtime.
# Invoked as: cmake -DMRW_SRC_DIR=<src> [-DMRW_SRC_FILES=<f;f>] -DMRW_LIB=<lib> -P banned_scan.cmake
# MRW_SRC_DIR globs *.c/*.h in a directory; MRW_SRC_FILES adds explicit files (for scanning a single
# allocation-free TU in a directory that also holds allocating ones, e.g. the bake core).
#
# (1) Source text scan: no allocation / threading / GPU calls in the scanned sources.
# (2) Best-effort binary symbol scan of the library (dumpbin/nm if available).

set(BANNED_CALLS
  malloc calloc realloc free aligned_alloc alloca _alloca _malloca _aligned_malloc
  pthread_create thrd_create _beginthread _beginthreadex CreateThread
  vkAllocateMemory vkCreateDevice glGenTextures glTexImage2D cudaMalloc)

set(SRCS "")
if(MRW_SRC_DIR)
  file(GLOB SRCS "${MRW_SRC_DIR}/*.c" "${MRW_SRC_DIR}/*.h")
endif()
if(MRW_SRC_FILES)
  list(APPEND SRCS ${MRW_SRC_FILES})
endif()
set(FOUND "")
foreach(f ${SRCS})
  file(READ "${f}" content)
  foreach(b ${BANNED_CALLS})
    string(REGEX MATCH "(^|[^A-Za-z0-9_])${b}[ \t\r\n]*\\(" m "${content}")
    if(m)
      list(APPEND FOUND "source ${f}: ${b}(")
    endif()
  endforeach()
  # C++ allocation / threading must never appear in C runtime sources either
  string(REGEX MATCH "(^|[^A-Za-z0-9_])(new|delete)[ \t\r\n]" mcpp "${content}")
  if(mcpp)
    list(APPEND FOUND "source ${f}: C++ new/delete")
  endif()
endforeach()

# (2) binary symbol scan - fatal if a banned import is actually referenced
if(MRW_LIB AND EXISTS "${MRW_LIB}")
  find_program(DUMPBIN dumpbin)
  find_program(NM nm)
  # CI (-DMRW_SCAN_STRICT=ON) refuses to silently skip the binary symbol scan when no tool is present;
  # locally it stays best-effort (the source-text scan above always runs and is the primary gate).
  if(MRW_SCAN_STRICT AND NOT DUMPBIN AND NOT NM)
    message(FATAL_ERROR "banned_scan (strict): no symbol tool (dumpbin/nm) for the binary symbol scan")
  endif()
  set(SYMS "")
  set(symrc 0)
  if(DUMPBIN)
    execute_process(COMMAND "${DUMPBIN}" /SYMBOLS "${MRW_LIB}" OUTPUT_VARIABLE SYMS RESULT_VARIABLE symrc ERROR_QUIET)
  elseif(NM)
    execute_process(COMMAND "${NM}" "${MRW_LIB}" OUTPUT_VARIABLE SYMS RESULT_VARIABLE symrc ERROR_QUIET)
  endif()
  # Under strict, a tool that exists but errors or yields no symbols is also a failure - otherwise the
  # binary scan could silently pass without ever inspecting a symbol.
  if(MRW_SCAN_STRICT AND (NOT "${symrc}" STREQUAL "0" OR SYMS STREQUAL ""))
    message(FATAL_ERROR "banned_scan (strict): symbol tool failed (rc=${symrc}) or produced no output for ${MRW_LIB}")
  endif()
  if(SYMS)
    foreach(b malloc calloc realloc free aligned_alloc _aligned_malloc pthread_create CreateThread)
      string(REGEX MATCH "[^A-Za-z0-9_]${b}[^A-Za-z0-9_]" sm "${SYMS}")
      if(sm)
        list(APPEND FOUND "symbol ${MRW_LIB}: ${b}")
      endif()
    endforeach()
  endif()
endif()

if(FOUND)
  string(REPLACE ";" "\n  " PRETTY "${FOUND}")
  message(FATAL_ERROR "banned constructs/symbols found:\n  ${PRETTY}")
endif()
message(STATUS "banned_scan: clean")
