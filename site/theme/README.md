# Vendored theme

The documentation site is styled with [Doxygen Awesome](https://github.com/jothepro/doxygen-awesome-css),
**v2.3.4**, vendored here so the docs build is hermetic (no network fetch, local == CI).

- `doxygen-awesome.css`, `doxygen-awesome-sidebar-only.css` — the theme (referenced by `Doxyfile`).
- `LICENSE` — Doxygen Awesome's MIT license.

To update: replace these files from the upstream tag and bump the version noted above.
Doxygen Awesome is MIT-licensed and independent of marrow's own MIT license.
