Result codes {#ref_errors}
============

Nearly every marrow function returns an `mrw_result`. `MRW_OK` (0) is success; everything else is a
non-fatal, typed rejection — marrow never aborts, asserts on caller error in release, or writes
partial output on a validation failure. Always check the return.

| Code | Value | Fires when |
|---|---|---|
| `MRW_OK` | 0 | Success. |
| `MRW_E_FORMAT` | 1 | Bad magic, version, endianness, flag, or a violated structural invariant — the blob is not a well-formed `.mrw`. |
| `MRW_E_RANGE` | 2 | An out-of-range index/offset/count, or a non-finite time/weight/pose value. On the pose and batch calls, a non-finite input yields this with **no output written**. |
| `MRW_E_ALIGN` | 3 | The blob base, an internal offset, or a caller buffer is misaligned (e.g. the blob is not ≥64-byte aligned, or batch scratch/output does not meet the required alignment). |
| `MRW_E_OVERFLOW` | 4 | Size/offset arithmetic would overflow — caught by the requirements queries and the loader rather than wrapping. |
| `MRW_E_CAPACITY` | 5 | A caller buffer is too small: `joint_count > joint_capacity`, or a batch output/scratch buffer smaller than the requirements query asked for. |
| `MRW_E_UNSUPPORTED` | 6 | A known but not-enabled feature — an unimplemented codec/encoding/section, a forced dispatch backend the host CPU can't run, or an IK chain whose transforms aren't a proper uniformly-scaled rotation. |
| `MRW_E_INCOMPATIBLE` | 7 | An identity mismatch or a missing section — e.g. a clip paired with the wrong skeleton, a baked section whose skeleton id doesn't match, or a typed view requested at a section of a different type. |

## Patterns

- **Loading:** treat any non-`MRW_OK` from `mrw_blob_open()` as "reject this file." After it returns
  `MRW_OK`, the accessors are checked locators and won't fault on valid indices.
- **Sizing:** the `*_requirements()` queries are overflow-checked; a `MRW_E_OVERFLOW` means the
  requested `joint_count × instance_count` is implausibly large.
- **Batch validation:** the dispatch is validated against its own feature bits (no silent downgrade),
  and non-finite `times[i]` is rejected before any output is written.

See @ref concept_conventions for the full out-param + result-code ABI contract.
