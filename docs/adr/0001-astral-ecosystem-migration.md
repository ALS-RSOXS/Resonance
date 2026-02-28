# ADR-001: Astral ecosystem migration

## Status

Accepted

## Context

The project needed to align with the modern Astral toolchain and layout prescribed by [copier-astral](https://github.com/ritwiktiwari/copier-astral): Hatch build, Ruff + ty, single installable package, tests, CI, pre-commit, and changelog. The codebase previously used setuptools, had two top-level packages (`api` and `mcp`) under `src/`, and used the legacy import name `api_dev` for the API package. There was no CI, no standard lint/format/type-check tooling, and no tests directory.

## Decision

- Adopt **Hatch** as the build backend and remove setuptools.
- Restructure to a **single installable package** `resonance` with subpackages `resonance.api` and `resonance.mcp` under `src/resonance/`.
- Fix **imports**: replace all `api_dev` references with `resonance.api` (or relative `..api` where appropriate).
- Add **Ruff** (lint + format) and **ty** (type check) configuration and dev dependencies; remove Ruff from main dependencies.
- Add **pytest** and **pytest-cov** with config; add a `tests/` directory with smoke tests.
- Add **Makefile** targets: install, verify, fix, lint, format, type-check, test, test-cov, pysentry.
- Add **pre-commit** (prek-style) with pre-commit-hooks, Ruff, and ty.
- Add **GitHub Actions** CI: verify, test, secret/dependency scan (Gitleaks, pysentry), and Semgrep SAST.
- Add **git-cliff** config and **CHANGELOG.md** for conventional-commit-based changelogs.
- Document this decision in an ADR under `docs/adr/`.

## Consequences

### Positive

- Single namespace and consistent imports (`from resonance.api.types import AI`).
- Build and quality tooling match the copier-astral template (Hatch, Ruff, ty, pytest).
- CI enforces lint, format, type-check, and tests on push/PR.
- Pre-commit keeps local commits aligned with CI.
- Changelog can be generated from conventional commits.

### Negative

- One-time migration cost: move of `api`/`mcp` into `resonance`, and import updates in four places.
- External consumers or notebooks that imported `api` or `mcp` must switch to `resonance.api` or `resonance.mcp`.

### Neutral

- Entry point remains `mcp-beamline`; implementation path becomes `resonance.mcp.__main__:main`.

## Alternatives Considered

- **Keep two top-level packages** (`src/api`, `src/mcp`) and only add Hatch + tooling: rejected to stay consistent with copier-astral’s single-package layout and to resolve `api_dev` naming in one go.
- **Use mypy instead of ty**: rejected in favor of ty to match the Astral template and reduce tool divergence.

## References

- [copier-astral](https://github.com/ritwiktiwari/copier-astral)
- [Hatch](https://hatch.pypa.io/)
- [Ruff](https://docs.astral.sh/ruff/)
- [ty](https://docs.astral.sh/ty/)
