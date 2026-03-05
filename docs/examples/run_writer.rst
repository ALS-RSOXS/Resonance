Write run data with RunWriter
=============================

Open a beamtime database, start a run and stream, write events and optional detector images, then close. The writer uses SQLite for metadata and a co-located Zarr store for images.

.. code-block:: python

   from pathlib import Path
   from resonance.api.data import RunWriter, SampleMetadata

   db_path = Path("beamtime.db")
   sample = SampleMetadata(name="PS", formula="C8H8")
   writer = RunWriter(db_path, sample)
   writer.open()

   run_uid = writer.open_run("grid_scan", metadata={"operator": "user"})
   stream_uid = writer.open_stream("primary", data_keys={"motor_x": {"dtype": "number"}})
   event_uid = writer.write_event({"motor_x": 1.0, "motor_y": 0.5})
   writer.close_run(exit_status="success")
   writer.close()

See :doc:`/api/data` for ``RunWriter``, ``SampleMetadata``, and the data schema.
