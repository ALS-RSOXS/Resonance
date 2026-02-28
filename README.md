# Resonance

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ALS-RSOXS/auto-reflect)

Resonance is a Python package for the ALS-RSOXS Beamline Control System. It provides a high-level API for motor control, analog inputs, NEXAFS scans, and an MCP server for beamline access.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
uv sync --all-groups
```

To include the optional BCS group (beamline control dependencies):

```bash
uv sync --all-groups --group bcs
```

## Usage

Run the MCP beamline server:

```bash
uv run mcp-beamline
```

Or after installing:

```bash
mcp-beamline
```

Use the API from Python:

```python
from resonance.api import RsoxsServer, nexafs_scan
from resonance.api.types import AI, Motor
```

## Development

```bash
make install
make verify
make test
```

- `make verify` runs lint, format-check, and type-check.
- `make fix` auto-fixes Ruff lint and format.
- `make test-cov` runs tests with coverage.

Install pre-commit hooks (prek or pre-commit):

```bash
pre-commit install
```

See [CHANGELOG.md](CHANGELOG.md) for release history. Architecture decisions are in [docs/adr/](docs/adr/).

## Project status

### Basic I/O operations

- [x] Read AIs
- [x] Read motor positions
- [x] Move motor positions
- [ ] Motor scan scheduling

### Scheduling and run management

- [ ] Build default NEXAFS scans
- [ ] Get feedback logic
- [ ] Build auto alignment

### QOL

- [x] Install BLS API into venv to avoid sys.path.append
- [x] Type motors and AIs
- [x] Formatting and type hinting (Ruff, ty)
