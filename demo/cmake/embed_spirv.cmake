# Convert a compiled SPIR-V binary into a C header holding a uint32 word array.
# Usage: cmake -DSPV=<in.spv> -DHDR=<out.h> -DSYM=<symbol> -P embed_spirv.cmake
#
# SPIR-V is a little-endian stream of 32-bit words; file(READ ... HEX) gives lowercase hex
# bytes in file order, so each word's bytes b0 b1 b2 b3 (b0 = LSB) become 0xb3b2b1b0.

file(READ "${SPV}" hexstr HEX)
string(LENGTH "${hexstr}" hexlen)
math(EXPR nbytes "${hexlen} / 2")
math(EXPR nwords "${nbytes} / 4")

set(body "")
set(col 0)
set(w 0)
while(w LESS nwords)
  math(EXPR off "${w} * 8")
  string(SUBSTRING "${hexstr}" ${off} 8 word)
  string(SUBSTRING "${word}" 0 2 b0)
  string(SUBSTRING "${word}" 2 2 b1)
  string(SUBSTRING "${word}" 4 2 b2)
  string(SUBSTRING "${word}" 6 2 b3)
  string(APPEND body "0x${b3}${b2}${b1}${b0}u,")
  math(EXPR col "${col} + 1")
  if(col EQUAL 8)
    string(APPEND body "\n    ")
    set(col 0)
  endif()
  math(EXPR w "${w} + 1")
endwhile()

file(WRITE "${HDR}"
"/* Generated from ${SPV} - do not edit. */\n#include <stdint.h>\nstatic const uint32_t ${SYM}[] = {\n    ${body}\n};\n")
