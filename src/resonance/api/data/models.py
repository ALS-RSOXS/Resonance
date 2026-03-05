from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampleMetadata:
    """In-memory representation of a sample row from the `samples` table.

    Parameters
    ----------
    name : str
        Human-readable sample name, used as the primary display identifier.
    formula : str | None
        Chemical formula string, e.g. "C8H8". None if not recorded.
    serial : str | None
        Physical identifier stamped on the sample or its holder, e.g. "S1234".
    tags : list[str]
        Arbitrary string tags for grouping or filtering, e.g. ["polymer", "reference"].
    beamline_pos : str | None
        Stage slot or carousel position where the sample was mounted.
    extra : dict[str, Any]
        Arbitrary metadata that does not fit the fixed schema.
    id : int | None
        Database primary key assigned on insertion. None before the row is written.

    Attributes
    ----------
    name : str
        Human-readable sample name, used as the primary display identifier.
    formula : str | None
        Chemical formula string, e.g. "C8H8". None if not recorded.
    serial : str | None
        Physical identifier stamped on the sample or its holder, e.g. "S1234".
    tags : list[str]
        Arbitrary string tags for grouping or filtering, e.g. ["polymer", "reference"].
    beamline_pos : str | None
        Stage slot or carousel position where the sample was mounted.
    extra : dict[str, Any]
        Arbitrary metadata that does not fit the fixed schema.
    id : int | None
        Database primary key assigned on insertion. None before the row is written.
    """

    name: str
    formula: str | None = None
    serial: str | None = None
    tags: list[str] = field(default_factory=list)
    beamline_pos: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    id: int | None = field(default=None)


@dataclass
class RunSummary:
    """Lightweight summary of a single run, used for catalog listing operations.

    This is a denormalized projection built from joining the `runs` and `samples`
    tables. It is intended for display and filtering, not for full data retrieval.

    Attributes
    ----------
    uid : str
        Run UUID in hexadecimal form, as stored in the database.
    plan_name : str
        Name of the scan plan that produced this run, e.g. "en_scan".
    time_start : float
        Unix timestamp marking the start of the run.
    sample_name : str | None
        Sample name denormalized from the `samples` join. None if no sample was
        associated with the run or if the join produced no match.
    time_stop : float | None
        Unix timestamp marking the end of the run. None if the run did not finish
        cleanly or if the stop document was not recorded.
    exit_status : str | None
        Final status string from the RunStop document. Expected values are
        "success", "aborted", or "failed". None if not recorded.
    """

    uid: str
    plan_name: str
    time_start: float
    sample_name: str | None = None
    time_stop: float | None = None
    exit_status: str | None = None


@dataclass
class BeamtimeInfo:
    """Represents a beamtime entry as stored in the master index database.

    Each beamtime corresponds to a single experimental session and owns its own
    SQLite database file. The master index holds one row per beamtime so that
    multiple sessions can be cataloged from a single entry point.

    Attributes
    ----------
    id : int
        Primary key in the master index database.
    researcher_name : str
        Name of the researcher who led the beamtime session.
    label : str
        Human-readable label for the beamtime, typically an ISO date string
        such as "2026-03-05".
    db_path : str
        Path to the beamtime SQLite file, relative to the DATA root directory.
    time_start : float | None
        Unix timestamp for the beginning of the beamtime. None if not recorded.
    time_stop : float | None
        Unix timestamp for the end of the beamtime. None if still in progress or
        not recorded.
    """

    id: int
    researcher_name: str
    label: str
    db_path: str
    time_start: float | None = None
    time_stop: float | None = None
