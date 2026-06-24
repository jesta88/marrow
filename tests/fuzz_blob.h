/* Shared entry for the .mrw loader fuzz target. One function, called by BOTH the libFuzzer target
 * (fuzz_blob.c's LLVMFuzzerTestOneInput) and the portable corpus replayer (fuzz_replay.c) - so the
 * replay path is a permanent regression gate even on toolchains without libFuzzer (e.g. MSVC). */
#ifndef MRW_FUZZ_BLOB_H
#define MRW_FUZZ_BLOB_H

#include <stddef.h>
#include <stdint.h>

/* Decode data[0..size) as a .mrw blob via mrw_blob_open; on MRW_OK, walk the full public read
 * surface (every accessor, at interior AND out-of-range boundary indices) so ASan/UBSan trap any
 * out-of-bounds read or UB the loader's validation missed. Returns 0 (libFuzzer convention). MUST
 * NOT crash on any input - that non-crash is exactly what the fuzzer is verifying. */
int mrw_fuzz_one(const uint8_t *data, size_t size);

#endif /* MRW_FUZZ_BLOB_H */
