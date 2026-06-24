The .mrw format at a glance {#ref_format}
===========================

`.mrw` is marrow's on-disk format: a byte-defined, little-endian container that the runtime loads by
**validating, then viewing** — no struct overlay, no `mmap + cast`. This page is a user-facing
orientation, not the byte-level normative specification; you only need it if you are writing your own
exporter or a binding. Most users produce `.mrw` files with `gltf2marrow` (see @ref howto_gltf) and
never touch the bytes.

[TOC]

## Shape

```
┌─────────────────────────────────────────────┐
│ header   magic "MRRW", version, endianness   │
│          section_count, section_table_off     │
├─────────────────────────────────────────────┤
│ section table   one entry per section:        │
│                 { type, offset, size, flags } │
├─────────────────────────────────────────────┤
│ section 0  (e.g. SKELETON)                    │
│ section 1  (e.g. CLIP)                        │
│ section 2  (e.g. BAKED)                        │
│ ...                                           │
└─────────────────────────────────────────────┘
```

The whole file is loaded into a single ≥64-byte-aligned buffer and validated in one deterministic
pass by `mrw_blob_open()`. Internal arrays are ≥16-byte aligned and addressed by section-relative
offsets, so the views can borrow the bytes directly.

## Section types

| Type | Holds |
|---|---|
| `MRW_SECTION_SKELETON` | The joint hierarchy: parents, rest-local transforms, inverse-bind matrices, joint names. A blob has at most one. |
| `MRW_SECTION_CLIP` | A dense animation clip: per-joint samples at a fixed `fps`, optional root-motion track, looping flag. |
| `MRW_SECTION_BAKED` | The Tier-B baked palette: per-bone `Q + T + scale` texels plus a clip table (see @ref concept_two_tiers). |

A section the reader doesn't recognize may be flagged `MRW_SECTION_FLAG_OPTIONAL` and skipped.

## Versioning and compatibility

The v0 contracts and the wire format are ratified and evolve **backward-additively**: new layouts
arrive only as *new* codec/encoding/section-type ids. An old reader dispatches on the id and fails
determinately on an unknown one; existing assets keep loading. Readers should always go through
`mrw_blob_open()` and the typed views rather than reading fields by offset.

## Content identities

Skeletons and clips carry a 128-bit FNV-1a-128 fingerprint over a layout-independent canonical
stream. The loader uses it to verify that a clip and skeleton (or a baked section and its skeleton)
belong together; a mismatch is `MRW_E_INCOMPATIBLE`. If you author a blob yourself, compute the id
with `mrw_skeleton_compute_id()` / `mrw_clip_compute_id()` and stamp it in (the id field is excluded
from its own stream).

## Safety

The loader is the trust boundary: truncated or corrupt input fails cleanly with a typed
`mrw_result` rather than reading out of bounds. It is coverage-guided fuzzed and runs clean under
ASan / UBSan / MSan.
