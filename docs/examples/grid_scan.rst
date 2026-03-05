Run a grid scan
===============

Minimal example: build a 2D grid scan plan and execute it with the beamline API. Requires a running BCS server and the BCS dependency group.

.. code-block:: python

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
       results = await bl.scan_from_dataframe(plan, progress=True)
       return results

   asyncio.run(main())

See :doc:`/api/utils` for ``create_grid_scan``, ``create_line_scan``, and ``create_energy_scan``, and :doc:`/api/scan` for ``ScanPlan`` and ``ScanExecutor``.
