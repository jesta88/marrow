# Vendored third-party code

These files are **not** part of the `marrow` runtime - the runtime is zero-dependency and never
links them. They are used only by the offline tool targets under `tools/` (gltf2marrow, marrow-bake).

## cgltf.h

- **Upstream:** https://github.com/jkuhlmann/cgltf
- **Version:** 1.15 (release tag `v1.15`)
- **Retrieved from:** https://raw.githubusercontent.com/jkuhlmann/cgltf/v1.15/cgltf.h
- **SHA-256:** `E378A21C084BF1F288BB799DE827BB26906EFB024255F1ECF1705EA13F11C6EC`
- **License:** MIT (Copyright (c) 2018-2021 Johannes Kuhlmann) - full text at the end of `cgltf.h`.

Single-header glTF 2.0 parser. Exactly one tool translation unit defines `CGLTF_IMPLEMENTATION`
before including it. Vendored verbatim; do not edit. To update, replace the file and refresh the
version/SHA above.
