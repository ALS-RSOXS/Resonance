Data models and writers
=======================

Beamtime data models, catalog, schema, and writers for SQLite and Zarr storage.

.. autoclass:: resonance.api.data.models.SampleMetadata
   :members:
   :show-inheritance:
   :exclude-members: name, formula, serial, tags, beamline_pos, extra, id

.. autoclass:: resonance.api.data.models.BeamtimeInfo
   :members:
   :show-inheritance:
   :exclude-members: db_path, id, label, researcher_name, time_start, time_stop

.. autoclass:: resonance.api.data.models.RunSummary
   :members:
   :show-inheritance:
   :exclude-members: uid, plan_name, time_start, sample_name, time_stop, exit_status

.. autoclass:: resonance.api.data.writer.RunWriter
   :members:
   :show-inheritance:

.. autoclass:: resonance.api.data.writer.IndexWriter
   :members:
   :show-inheritance:

.. autoclass:: resonance.api.data.catalog.Catalog
   :members:
   :show-inheritance:

.. autoclass:: resonance.api.data.catalog.Run
   :members:
   :show-inheritance:

.. autoclass:: resonance.api.data.catalog.LazyImageSequence
   :members:
   :show-inheritance:

Schema helpers:

.. automodule:: resonance.api.data.schema
   :members:
   :undoc-members:
