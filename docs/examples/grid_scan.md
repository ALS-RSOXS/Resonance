# Run a Grid Scan

Build a 2D grid scan plan and execute it with the beamline API.

```python
import asyncio
from resonance.api import Beamline, ScanPlan, create_grid_scan


async def main():
    bl = await Beamline.create()
    grid_df = create_grid_scan(
        x_range=(0.0, 2.0, 3),
        y_range=(0.0, 1.0, 2),
        exposure_time=0.1,
    )
    plan = ScanPlan.from_dataframe(grid_df, ai_channels=["Photodiode"])
    return await bl.scan_from_dataframe(plan, progress=True)


asyncio.run(main())
```

Related references:

- [API Reference](../api.md)
