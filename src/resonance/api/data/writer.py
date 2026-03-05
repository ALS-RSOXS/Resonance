from __future__ import annotations

import json
import sqlite3
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from resonance.api.data.models import SampleMetadata
from resonance.api.data.schema import create_beamtime_schema

if TYPE_CHECKING:
    from pathlib import Path


class RunWriter:
    """Manages a SQLite connection to a beamtime database and writes scan data.

    Parameters
    ----------
    db_path : Path
        Path to the beamtime SQLite database file. Created on first open if it
        does not exist.
    sample : SampleMetadata
        Sample to associate with runs written through this writer. If
        ``sample.id`` is None, the sample is upserted on ``open``.

    Attributes
    ----------
    _db_path : Path
        Resolved path to the SQLite file.
    _sample : SampleMetadata
        Sample metadata, with ``id`` populated after ``open``.
    _conn : sqlite3.Connection or None
        Active database connection, or None when closed.
    _run_uid : str
        Hex UID of the currently open run, or empty string.
    _stream_uid : str
        Hex UID of the currently open stream, or empty string.
    _seq_num : int
        Event sequence counter within the current stream.
    """

    def __init__(self, db_path: Path, sample: SampleMetadata) -> None:
        self._db_path = db_path
        self._sample = sample
        self._conn: sqlite3.Connection | None = None
        self._run_uid: str = ""
        self._stream_uid: str = ""
        self._seq_num: int = 0

    def open(self) -> None:
        """Open the database connection and upsert the sample.

        Creates the beamtime schema if the database does not yet exist. If
        ``self._sample.id`` is None, the sample is looked up by name; if a
        matching row exists its id is loaded, otherwise a new row is inserted.

        Raises
        ------
        RuntimeError
            If the writer is already open (``self._conn`` is not None).
        sqlite3.DatabaseError
            If the database file is corrupt or unreadable.
        """
        if self._conn is not None:
            raise RuntimeError("RunWriter is already open")
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        create_beamtime_schema(conn)
        if self._sample.id is None:
            row = conn.execute(
                "SELECT id FROM samples WHERE name = ?", (self._sample.name,)
            ).fetchone()
            if row is not None:
                self._sample.id = row[0]
            else:
                cur = conn.execute(
                    "INSERT INTO samples (name, formula, serial, tags, beamline_pos, extra) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        self._sample.name,
                        self._sample.formula,
                        self._sample.serial,
                        json.dumps(self._sample.tags),
                        self._sample.beamline_pos,
                        json.dumps(self._sample.extra),
                    ),
                )
                conn.commit()
                self._sample.id = cur.lastrowid
        self._conn = conn

    def open_run(self, plan_name: str, *, metadata: dict[str, Any] | None = None) -> str:
        """Insert a new run row and return its UID.

        Parameters
        ----------
        plan_name : str
            Name of the scan plan, e.g. "en_scan".
        metadata : dict[str, Any] or None, optional
            Arbitrary run-level metadata serialized to JSON. Defaults to an
            empty dict.

        Returns
        -------
        str
            Hex UUID of the newly created run.

        Raises
        ------
        RuntimeError
            If the writer is not open.
        """
        if self._conn is None:
            raise RuntimeError("RunWriter is not open")
        uid = uuid4().hex
        self._run_uid = uid
        self._seq_num = 0
        self._conn.execute(
            "INSERT INTO runs (uid, sample_id, plan_name, time_start, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (uid, self._sample.id, plan_name, time.time(), json.dumps(metadata or {})),
        )
        self._conn.commit()
        return uid

    def open_stream(self, name: str, data_keys: dict[str, Any]) -> str:
        """Insert a new stream row and return its UID.

        Parameters
        ----------
        name : str
            Stream name, e.g. "primary".
        data_keys : dict[str, Any]
            Descriptor mapping field names to their metadata, serialized to JSON.

        Returns
        -------
        str
            Hex UUID of the newly created stream.

        Raises
        ------
        RuntimeError
            If the writer is not open or no run has been opened.
        """
        if self._conn is None:
            raise RuntimeError("RunWriter is not open")
        if not self._run_uid:
            raise RuntimeError("No open run")
        uid = uuid4().hex
        self._stream_uid = uid
        self._conn.execute(
            "INSERT INTO streams (uid, run_uid, name, data_keys, time_created) "
            "VALUES (?, ?, ?, ?, ?)",
            (uid, self._run_uid, name, json.dumps(data_keys), time.time()),
        )
        self._conn.commit()
        return uid

    def write_event(
        self,
        data: dict[str, float | int | str | bool],
        timestamps: dict[str, float] | None = None,
    ) -> str:
        """Insert an event row and return its UID.

        Events are not committed individually; the commit is deferred to
        ``close_run`` for performance.

        Parameters
        ----------
        data : dict[str, float | int | str | bool]
            Measured values keyed by field name.
        timestamps : dict[str, float] or None, optional
            Per-field acquisition timestamps. Defaults to an empty dict.

        Returns
        -------
        str
            Hex UUID of the newly inserted event.

        Raises
        ------
        RuntimeError
            If the writer is not open, no run has been opened, or no stream
            has been opened.
        """
        if self._conn is None:
            raise RuntimeError("RunWriter is not open")
        if not self._run_uid:
            raise RuntimeError("No open run")
        if not self._stream_uid:
            raise RuntimeError("No open stream")
        self._seq_num += 1
        uid = uuid4().hex
        self._conn.execute(
            "INSERT INTO events (uid, stream_uid, seq_num, time, data, timestamps) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                uid,
                self._stream_uid,
                self._seq_num,
                time.time(),
                json.dumps(data),
                json.dumps(timestamps or {}),
            ),
        )
        return uid

    def close_run(self, *, exit_status: str = "success") -> None:
        """Finalize the current run and commit all pending events.

        Parameters
        ----------
        exit_status : str, optional
            Final status string. Expected values are "success", "aborted", or
            "failed". Defaults to "success".

        Raises
        ------
        RuntimeError
            If the writer is not open or no run has been opened.
        """
        if self._conn is None:
            raise RuntimeError("RunWriter is not open")
        if not self._run_uid:
            raise RuntimeError("No open run")
        self._conn.execute(
            "UPDATE runs SET time_stop = ?, exit_status = ?, num_events = ? "
            "WHERE uid = ?",
            (time.time(), exit_status, self._seq_num, self._run_uid),
        )
        self._conn.commit()

    def close(self) -> None:
        """Commit any remaining work and close the database connection.

        Raises
        ------
        RuntimeError
            If the writer is not open.
        """
        if self._conn is None:
            raise RuntimeError("RunWriter is not open")
        self._conn.commit()
        self._conn.close()
        self._conn = None

    def __enter__(self) -> RunWriter:
        """Open the writer and return self.

        Returns
        -------
        RunWriter
            The opened writer instance.
        """
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Close the run and connection, propagating any exception.

        Parameters
        ----------
        exc_type : type[BaseException] or None
            Exception type if an exception occurred, otherwise None.
        exc : BaseException or None
            Exception instance if an exception occurred, otherwise None.
        tb : object
            Traceback object if an exception occurred, otherwise None.
        """
        if exc_type is not None:
            if self._run_uid:
                self.close_run(exit_status="failed")
            self.close()
        elif self._run_uid:
            self.close_run()
            self.close()


class IndexWriter:
    """Writer for the master index database aggregating cross-beamtime metadata.

    Parameters
    ----------
    index_db_path : Path
        Path to the master index SQLite database file.

    Notes
    -----
    This class is a skeleton. All methods raise ``NotImplementedError`` until
    the master index feature is implemented.
    """

    def __init__(self, index_db_path: Path) -> None:
        self._index_db_path = index_db_path

    def ensure_schema(self) -> None:
        """Create the index schema if it does not already exist.

        Raises
        ------
        NotImplementedError
            Always; not yet implemented.
        """
        raise NotImplementedError("IndexWriter is not yet implemented")

    def register_researcher(
        self,
        name: str,
        root_path: str,
        *,
        email: str | None = None,
        orcid: str | None = None,
        affiliation: str | None = None,
    ) -> int:
        """Insert or retrieve a researcher row and return its id.

        Parameters
        ----------
        name : str
            Full name of the researcher.
        root_path : str
            Filesystem root under which this researcher's data lives.
        email : str or None, optional
            Contact email address.
        orcid : str or None, optional
            ORCID identifier string.
        affiliation : str or None, optional
            Institutional affiliation.

        Returns
        -------
        int
            Primary key of the researcher row.

        Raises
        ------
        NotImplementedError
            Always; not yet implemented.
        """
        raise NotImplementedError("IndexWriter is not yet implemented")

    def register_beamtime(
        self,
        researcher_id: int,
        label: str,
        db_path: str,
        *,
        time_start: float | None = None,
    ) -> int:
        """Insert or retrieve a beamtime row and return its id.

        Parameters
        ----------
        researcher_id : int
            Foreign key into the ``researchers`` table.
        label : str
            Human-readable beamtime label, e.g. "2026-03-05".
        db_path : str
            Path to the per-beamtime SQLite file.
        time_start : float or None, optional
            Unix timestamp for the start of the beamtime session.

        Returns
        -------
        int
            Primary key of the beamtime row.

        Raises
        ------
        NotImplementedError
            Always; not yet implemented.
        """
        raise NotImplementedError("IndexWriter is not yet implemented")

    def index_run(
        self,
        uid: str,
        researcher_id: int,
        beamtime_id: int,
        plan_name: str,
        *,
        sample_name: str | None = None,
        time_start: float | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Insert a run summary row into the master index.

        Parameters
        ----------
        uid : str
            Hex UUID of the run, matching the per-beamtime ``runs.uid``.
        researcher_id : int
            Foreign key into the ``researchers`` table.
        beamtime_id : int
            Foreign key into the ``beamtimes`` table.
        plan_name : str
            Name of the scan plan.
        sample_name : str or None, optional
            Sample name denormalized from the per-beamtime database.
        time_start : float or None, optional
            Unix timestamp for the start of the run.
        tags : list[str] or None, optional
            Arbitrary string tags for filtering.

        Raises
        ------
        NotImplementedError
            Always; not yet implemented.
        """
        raise NotImplementedError("IndexWriter is not yet implemented")
