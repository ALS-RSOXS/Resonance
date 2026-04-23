# Quick Start

## Run the MCP beamline server

From the project root with the BCS group installed:

```bash
uv run mcp-beamline
```

Or after installing the package:

```bash
mcp-beamline
```

## Use the API from Python

Connect to the beamline and run a DataFrame-driven scan.

```python
import asyncio
import pandas as pd
from resonance.api import Beamline, ScanPlan


async def main():
    bl = await Beamline.create()
    df = pd.DataFrame(
        {
            "Sample X": [0.0, 1.0, 2.0],
            "exposure": [0.1, 0.1, 0.1],
        }
    )
    plan = ScanPlan.from_dataframe(df, ai_channels=["Photodiode"])
    return await bl.scan_from_dataframe(plan, progress=True)


asyncio.run(main())
```

See API pages for details:

- [Beamline](../api/beamline.md)
- [Scan](../api/scan.md)
- [AI accessor](../api/ai.md)
- [Motor accessor](../api/motors.md)
