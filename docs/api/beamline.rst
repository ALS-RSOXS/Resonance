Beamline and Connection
=======================

The ``Beamline`` class is the high-level facade for beamline hardware control. ``Connection`` holds BCS server settings from the environment.

.. autoclass:: resonance.api.core.beamline.Connection
   :members:
   :show-inheritance:

.. autoclass:: resonance.api.core.beamline.Beamline
   :members:
   :show-inheritance:

Examples
--------

Creating a beamline from environment variables (``BCS_SERVER_ADDRESS``, ``BCS_SERVER_PORT``):

.. code-block:: python

   bl = await Beamline.create()
   data = await bl.ai.trigger_and_read(["Photodiode"], acquisition_time=1.0)
   await bl.motors.set("Sample X", 10.5)
