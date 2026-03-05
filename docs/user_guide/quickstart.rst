Quick start
===========

Run the MCP beamline server
---------------------------

From the project root (with BCS group installed):

.. code-block:: bash

   uv run mcp-beamline

Or after installing the package:

.. code-block:: bash

   mcp-beamline

Use the API from Python
-----------------------

Connect to the beamline and run a DataFrame-driven scan. You need a BCS server running and the optional BCS dependency installed.

.. code-block:: python

   import asyncio
   import pandas as pd
   from resonance.api import Beamline, ScanPlan, ScanExecutor
   from resonance.api.types import Motor, AI

   async def main():
       bl = await Beamline.create()
       df = pd.DataFrame({
           "Sample X": [0.0, 1.0, 2.0],
           "exposure": [0.1, 0.1, 0.1],
       })
       plan = ScanPlan.from_dataframe(df, ai_channels=["Photodiode"])
       results = await bl.scan_from_dataframe(plan, progress=True)
       return results

   asyncio.run(main())

For full parameters and behavior see the API Reference:

* :doc:`/api/beamline` for ``Beamline`` and ``Connection``
* :doc:`/api/scan` for ``ScanPlan`` and ``ScanExecutor``
* :doc:`/api/ai` for analog input access
* :doc:`/api/motors` for motor access
