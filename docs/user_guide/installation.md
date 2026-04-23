# Installation

## Requirements

- [uv](https://docs.astral.sh/uv/) for package and dependency management
- Python 3.13 or newer

## Basic installation

From the project root:

```bash
uv sync --all-groups
```

This installs the core package and development tools.

## Optional BCS group

To use the Beamline API and hardware control, install the optional BCS group:

```bash
uv sync --all-groups --group bcs
```

## Verify installation

Run quality checks:

```bash
make verify
```

Run tests:

```bash
make test
```

## Development hooks

Install hooks so checks run before commit:

```bash
prek install
```
