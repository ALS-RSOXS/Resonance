from __future__ import annotations

import json
import sqlite3
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

import pandas as pd

from resonance.api.data.models import RunSummary, SampleMetadata

if TYPE_CHECKING:
    import numpy as np

_SQL_RECENT = """
SELECT r.uid, r.plan_name, r.time_start, r.time_stop, r.exit_status,
       s.name AS sample_name
FROM runs r
LEFT JOIN samples s ON r.sample_id = s.id
ORDER BY r.time_start DESC
LIMIT ?
"""

_SQL_BY_SAMPLE = """
SELECT r.uid, r.plan_name, r.time_start, r.time_stop, r.exit_status,
       s.name AS sample_name
FROM runs r
JOIN samples s ON r.sample_id = s.id
WHERE s.name = ?
ORDER BY r.time_start DESC
"""

_SQL_RUN_BY_UID = "SELECT * FROM runs WHERE uid = ?"

_SQL_SAMPLE_BY_ID = "SELECT * FROM samples WHERE id = ?"

_SQL_EVENTS = """
SELECT e.seq_num, e.time, e.data
FROM events e
JOIN streams s ON e.stream_uid = s.uid
WHERE s.run_uid = ? AND s.name = ?
ORDER BY e.seq_num
"""


def _row_to_run_summary(row: sqlite3.Row) -> RunSummary:
    return RunSummary(
        uid=row["uid"],
        plan_name=row["plan_name"],
        time_start=row["time_start"],
        time_stop=row["time_stop"],
        exit_status=row["exit_status"],
        sample_name=row["sample_name"],
    )


class Catalog:
    """Read-only catalog over a per-beamtime SQLite database.

    Parameters
    ----------
    db_path : Path
        Path to the beamtime SQLite database file.

    Attributes
    ----------
    _conn : sqlite3.Connection
        Read-only connection to the database.
    _db_path : Path
        Resolved path passed at construction time.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row

    def recent(self, n: int = 10) -> list[RunSummary]:
        """Return the n most recently started runs.

        Parameters
        ----------
        n : int, optional
            Maximum number of runs to return. Defaults to 10.

        Returns
        -------
        list[RunSummary]
            Run summaries ordered newest-first.
        """
        rows = self._conn.execute(_SQL_RECENT, (n,)).fetchall()
        return [_row_to_run_summary(r) for r in rows]

    def by_sample(self, name: str) -> list[RunSummary]:
        """Return all runs associated with a sample by name.

        Parameters
        ----------
        name : str
            Exact sample name as stored in the ``samples`` table.

        Returns
        -------
        list[RunSummary]
            Run summaries for the given sample, ordered newest-first.
        """
        rows = self._conn.execute(_SQL_BY_SAMPLE, (name,)).fetchall()
        return [_row_to_run_summary(r) for r in rows]

    def __getitem__(self, uid: str) -> Run:
        """Return the full Run object for the given UID.

        Parameters
        ----------
        uid : str
            Hex UUID of the run.

        Returns
        -------
        Run
            Fully-featured run accessor backed by the open connection.

        Raises
        ------
        KeyError
            If no run with the given UID exists in the database.
        """
        row = self._conn.execute(_SQL_RUN_BY_UID, (uid,)).fetchone()
        if row is None:
            raise KeyError(uid)
        return Run(self._conn, dict(row))

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        self.close()


class Run:
    """Full accessor for a single run, including scalar event data and sample metadata.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active connection to the beamtime database.
    row : dict[str, Any]
        Deserialized row from the ``runs`` table.

    Attributes
    ----------
    _conn : sqlite3.Connection
        Shared database connection from the parent Catalog.
    _row : dict[str, Any]
        Raw column values for this run.
    """

    def __init__(self, conn: sqlite3.Connection, row: dict[str, Any]) -> None:
        self._conn = conn
        self._row = row

    @property
    def uid(self) -> str:
        """str: Hex UUID of the run."""
        return self._row["uid"]

    @property
    def plan_name(self) -> str:
        """str: Name of the scan plan."""
        return self._row["plan_name"]

    @property
    def time_start(self) -> float:
        """float: Unix timestamp of run start."""
        return self._row["time_start"]

    @property
    def time_stop(self) -> float | None:
        """float or None: Unix timestamp of run stop."""
        return self._row["time_stop"]

    @property
    def exit_status(self) -> str | None:
        """str or None: Final run status, e.g. 'success', 'failed'."""
        return self._row["exit_status"]

    @property
    def num_events(self) -> int:
        """int: Total number of events recorded in the primary stream."""
        return self._row["num_events"]

    @property
    def sample(self) -> SampleMetadata | None:
        """Return the associated sample metadata, or None if not set.

        Returns
        -------
        SampleMetadata or None
            Populated from the ``samples`` table row, with ``tags`` and
            ``extra`` deserialized from JSON. Returns None if ``sample_id``
            is null or the referenced row does not exist.
        """
        sample_id = self._row.get("sample_id")
        if sample_id is None:
            return None
        row = self._conn.execute(_SQL_SAMPLE_BY_ID, (sample_id,)).fetchone()
        if row is None:
            return None
        return SampleMetadata(
            name=row["name"],
            formula=row["formula"],
            serial=row["serial"],
            tags=json.loads(row["tags"] or "[]"),
            beamline_pos=row["beamline_pos"],
            extra=json.loads(row["extra"] or "{}"),
            id=row["id"],
        )

    def table(self, stream: str = "primary") -> pd.DataFrame:
        """Load scalar event data from a named stream as a DataFrame.

        Parameters
        ----------
        stream : str, optional
            Name of the stream to load. Defaults to "primary".

        Returns
        -------
        pd.DataFrame
            Columns are ``seq_num``, ``time``, followed by all scalar fields
            present in the event ``data`` JSON. Returns an empty DataFrame
            with columns ``["seq_num", "time"]`` if the stream has no events.
        """
        rows = self._conn.execute(_SQL_EVENTS, (self._row["uid"], stream)).fetchall()
        if not rows:
            return pd.DataFrame(columns=pd.Index(["seq_num", "time"]))
        records: list[dict[str, Any]] = []
        for row in rows:
            record: dict[str, Any] = {"seq_num": row["seq_num"], "time": row["time"]}
            record.update(json.loads(row["data"]))
            records.append(record)
        df = pd.DataFrame(records)
        leading = ["seq_num", "time"]
        remaining = [c for c in df.columns if c not in leading]
        return pd.DataFrame(df[leading + remaining])


class LazyImageSequence:
    """Lazy accessor for detector images referenced via Zarr.

    Images are not loaded until explicitly indexed. Each image
    is stored as a frame in a Zarr array on the filesystem;
    this class holds the reference metadata and defers loading.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active connection to the beamtime database.
    refs : list[dict[str, Any]]
        List of deserialized rows from the ``image_refs`` table.

    Attributes
    ----------
    _conn : sqlite3.Connection
        Shared database connection from the parent Catalog.
    _refs : list[dict[str, Any]]
        Image reference metadata, one entry per frame.
    """

    def __init__(self, conn: sqlite3.Connection, refs: list[dict[str, Any]]) -> None:
        self._conn = conn
        self._refs = refs

    def __len__(self) -> int:
        return len(self._refs)

    @property
    def shape(self) -> tuple[int, int, int]:
        """tuple[int, int, int]: (n_frames, shape_x, shape_y)."""
        if not self._refs:
            return (0, 0, 0)
        return (len(self._refs), self._refs[0]["shape_x"], self._refs[0]["shape_y"])

    def __getitem__(self, idx: int | slice) -> np.ndarray:
        """Raises NotImplementedError until Zarr detector integration is added.

        Parameters
        ----------
        idx : int or slice
            Frame index or slice.

        Raises
        ------
        NotImplementedError
            Always; Zarr-backed image loading is not yet implemented.
        """
        raise NotImplementedError(
            "Image loading from Zarr is not yet implemented. "
            "Detector integration is pending."
        )
