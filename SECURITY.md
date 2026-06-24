# Security Policy

marrow is a parsing library: it loads binary `.mrw` blobs that may come from untrusted sources. The
loader (`mrw_blob_open` and the validators it calls) is the security boundary — it **validates, then
views**, and is exercised by a coverage-guided libFuzzer target plus ASan/UBSan/MSan builds in CI.
Reports against that boundary are especially welcome.

## Supported versions

marrow is pre-1.0; security fixes land on `master` and in the next release. There is no
long-term-support branch yet.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for an undisclosed
vulnerability.

- **Preferred:** open a private report via GitHub's **Security → Report a vulnerability**
  ([Private vulnerability reporting](https://github.com/jesta88/marrow/security/advisories/new)) on
  this repository.
- **Alternatively:** email **jeremie.stamand@gmail.com**.

Please include the marrow version/commit, the platform and compiler, and steps to reproduce. A
crashing `.mrw` input under ASan/UBSan/MSan, or a libFuzzer reproducer, is ideal.

You can expect an acknowledgement within a few days. Once a fix is ready we will coordinate disclosure
and credit you in the release notes, unless you prefer to remain anonymous.

## Scope

**In scope:** memory-safety or undefined-behavior bugs reachable through `mrw_blob_open` and the
public read surface on attacker-controlled input — integer overflow, out-of-bounds reads, or unchecked
offsets in the loader or samplers.

**Out of scope:** the offline tools (`gltf2marrow`, `marrow-bake`) and the demo are build-time /
developer tools, not part of the zero-dependency runtime; issues there are ordinary bugs, not runtime
vulnerabilities. The `cgltf` reader vendored by `gltf2marrow` is third-party.
