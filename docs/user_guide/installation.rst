Installation
============

Requirements
------------

* `uv <https://docs.astral.sh/uv/>`_ for package and dependency management
* Python 3.13 or newer

Basic installation
------------------

From the project root:

.. code-block:: bash

   uv sync --all-groups

This installs the core package and dev tools (pytest, ruff, ty).

Optional: BCS (beamline control)
--------------------------------

To use the Beamline API and hardware control, install the optional BCS group:

.. code-block:: bash

   uv sync --all-groups --group bcs

Verifying installation
----------------------

Run lint, format-check, and type-check:

.. code-block:: bash

   make verify

Run tests:

.. code-block:: bash

   make test

For development
---------------

Install pre-commit hooks so lint and format run before each commit:

.. code-block:: bash

   pre-commit install

See the API Reference section (e.g. :doc:`/api/beamline`, :doc:`/api/scan`) for the full API. For a minimal usage example, see :doc:`quickstart`.
