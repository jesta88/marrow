/* The single translation unit that compiles cgltf's implementation (cgltf.h "Building": define
 * CGLTF_IMPLEMENTATION in exactly one source file). Isolating it here keeps the ~9k-line parser
 * out of every rebuild of the converter and confines the vendored loader's warnings to one object
 * (compiled at a lowered warning level - we do not patch vendored code). */
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
