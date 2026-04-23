## Learned User Preferences

- Wants this repository's documentation to follow the same MkDocs setup, assets, stylesheets, and JavaScript patterns as the sibling `refloxide` project, and to prefer markdown under `docs/` with tooling aligned to that pattern rather than a separate Sphinx-style scaffold in `docs/`.
- Prefers not to keep Cursor hook state and shared continual-learning tracking files in the repository when they get in the way of collaboration or a clean working tree.

## Learned Workspace Facts

- The documentation site is MkDocs; a standard local build check is `make docs`.
- Beamline end-user pages live under `docs/beamline/` and are linked from a `Beamline` block in `mkdocs.yml` (content derived in the same spirit as the xray-pro beamline reference, not as a dump of BCS command tables).
- The repository contains a Python BCS client and the `bcs-rs` crate; the intended direction is Rust-backed I/O and request handling (PyO3) with Python holding thin BCS request wrappers, including eventual migration of shared concepts like `MotorStatus` into Rust.
