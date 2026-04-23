# resonance

A Python beamline control toolkit for ALS-RSOXS workflows with typed device access, DataFrame-driven scans, and MCP integration.

## Installation

Install with `uv`:

```bash
uv sync --all-groups
```

Include optional beamline control dependencies:

```bash
uv sync --all-groups --group bcs
```

## Quick Start

```python
from resonance.api import Beamline, ScanPlan

bl = await Beamline.create()
plan = ScanPlan.from_dataframe(scan_df, ai_channels=["Photodiode"])
results = await bl.scan_from_dataframe(plan, progress=True)
```

## Team and support

- ALS-REIXS team: [RIXS Program at ALS](https://als.lbl.gov/science/photon-science-programs/rixs-program/)
- Group GitHub: [ALS-RSOXS organization](https://github.com/ALS-RSOXS)
- Submit issues: [auto-reflect issue tracker](https://github.com/ALS-RSOXS/auto-reflect/issues)

## Development

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

### Setup

```bash
git clone https://github.com/ALS-RSOXS/auto-reflect.git
cd auto-reflect
uv sync --all-groups
```

### Quality checks

```bash
make lint
make type-check
make test
```

### Build docs

```bash
make docs
```
