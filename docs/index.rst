.. resonance documentation master file

Welcome to resonance's documentation!
=====================================

**resonance** is a Python package for the ALS-RSOXS Beamline Control System. It provides a high-level API for motor control, analog inputs, DataFrame-driven scans, and an MCP server for beamline access.

Quick Start
-----------

.. code-block:: python

   from resonance.api import Beamline, ScanPlan, create_line_scan
   from resonance.api.types import Motor, AI

   # Connect to beamline (requires BCS server)
   bl = await Beamline.create()
   # Run a scan from a DataFrame
   plan = ScanPlan.from_dataframe(scan_df, ai_channels=["Photodiode"])
   results = await bl.scan_from_dataframe(plan, progress=True)

Installation
------------

Requires `uv <https://docs.astral.sh/uv/>`_ and Python 3.13+.

.. code-block:: bash

   uv sync --all-groups

To include the optional BCS group (beamline control dependencies):

.. code-block:: bash

   uv sync --all-groups --group bcs

To build documentation:

.. code-block:: bash

   uv sync --optional docs
   cd docs && make html

Features
--------

* **Beamline facade**: High-level interface (scan_from_dataframe, abort_scan, is_scanning)
* **Typed accessors**: MotorAccessor, AIAccessor, DIOAccessor for hardware I/O
* **DataFrame-driven scans**: ScanPlan and ScanExecutor for run orchestration
* **Data writing**: RunWriter for SQLite + Zarr beamtime storage
* **MCP server**: ``mcp-beamline`` for remote beamline access

User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/beamline
   api/scan
   api/det
   api/motors
   api/ai
   api/dio
   api/data
   api/types
   api/utils
   api/mcp

Examples
--------

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/grid_scan
   examples/run_writer

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
