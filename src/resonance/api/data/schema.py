from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import sqlite3

BEAMTIME_SCHEMA_VERSION: Final[int] = 1

_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS samples (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    formula      TEXT,
    serial       TEXT,
    tags         TEXT,
    beamline_pos TEXT,
    extra        TEXT
);

CREATE INDEX IF NOT EXISTS idx_samples_name ON samples(name);

CREATE TABLE IF NOT EXISTS runs (
    uid          TEXT PRIMARY KEY,
    sample_id    INTEGER REFERENCES samples(id) ON DELETE SET NULL,
    plan_name    TEXT NOT NULL,
    time_start   REAL NOT NULL,
    time_stop    REAL,
    exit_status  TEXT,
    num_events   INTEGER DEFAULT 0,
    operator     TEXT,
    metadata     TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_sample ON runs(sample_id);
CREATE INDEX IF NOT EXISTS idx_runs_plan   ON runs(plan_name);
CREATE INDEX IF NOT EXISTS idx_runs_tstart ON runs(time_start);

CREATE TABLE IF NOT EXISTS streams (
    uid          TEXT PRIMARY KEY,
    run_uid      TEXT NOT NULL REFERENCES runs(uid) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    data_keys    TEXT NOT NULL,
    time_created REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_streams_run ON streams(run_uid);

CREATE TABLE IF NOT EXISTS events (
    uid        TEXT PRIMARY KEY,
    stream_uid TEXT NOT NULL REFERENCES streams(uid) ON DELETE CASCADE,
    seq_num    INTEGER NOT NULL,
    time       REAL NOT NULL,
    data       TEXT NOT NULL,
    timestamps TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_stream ON events(stream_uid);
CREATE INDEX IF NOT EXISTS idx_events_seq    ON events(stream_uid, seq_num);

CREATE TABLE IF NOT EXISTS image_refs (
    -- TODO: add compression_codec column when detector integration is added
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid      TEXT NOT NULL REFERENCES events(uid) ON DELETE CASCADE,
    field_name     TEXT NOT NULL,
    zarr_group     TEXT NOT NULL,
    index_in_stack INTEGER NOT NULL,
    shape_x        INTEGER NOT NULL,
    shape_y        INTEGER NOT NULL,
    dtype          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_image_refs_event ON image_refs(event_uid);
"""


def create_beamtime_schema(conn: sqlite3.Connection) -> None:
    """
    Create all per-beamtime tables, indexes, and pragmas.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to the beamtime SQLite database file.

    Raises
    ------
    sqlite3.DatabaseError
        If DDL execution fails due to a malformed database or I/O error.

    Notes
    -----
    Each beamtime session is stored in its own SQLite `.db` file alongside
    a Zarr store directory. The `image_refs` table holds references into
    the Zarr array (group path and frame index) rather than binary BLOBs,
    keeping the SQLite file small and enabling memory-mapped array access.
    Foreign keys are enforced on every connection via ``PRAGMA foreign_keys = ON``,
    which must be re-applied per connection because SQLite resets it on open.
    """
    conn.executescript(_DDL)
    conn.execute(f"PRAGMA user_version = {BEAMTIME_SCHEMA_VERSION}")
    conn.commit()


def migrate_beamtime_schema(
    conn: sqlite3.Connection,
    target_version: int = BEAMTIME_SCHEMA_VERSION,
) -> None:
    """
    Apply pending schema migrations up to target_version.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to the beamtime SQLite database file.
    target_version : int, optional
        Schema version to migrate to. Defaults to ``BEAMTIME_SCHEMA_VERSION``.

    Raises
    ------
    RuntimeError
        If the database's current ``user_version`` is greater than
        ``target_version``, indicating a downgrade attempt or a database
        written by a newer version of this software.

    Notes
    -----
    Migration steps are applied sequentially from current_version + 1 up to
    target_version. Version 1 is the initial schema; future versions will add
    individual ``ALTER TABLE`` or data-transform statements here.
    """
    (current_version,) = conn.execute("PRAGMA user_version").fetchone()
    if current_version == target_version:
        return
    if current_version > target_version:
        raise RuntimeError(
            f"Database schema version {current_version} is newer than "
            f"target version {target_version}. Downgrade is not supported."
        )
