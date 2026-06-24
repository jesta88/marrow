## Summary

<!-- What does this change do, and why? -->

## Changes

<!-- Bullet the notable changes. -->

## Verification

<!-- How did you test this? -->
- [ ] `ctest --test-dir <build> --output-on-failure` passes locally
- [ ] New/changed behavior has test coverage (SIMD → scalar parity; loader → malformed + corpus)
- [ ] No new runtime dependency, allocation, threading, or graphics-API call
- [ ] Public API / `.mrw` format changes are backward-additive and reflected in `marrow.h`

## Notes

<!-- Trade-offs, follow-ups, open questions reviewers should know about. -->
