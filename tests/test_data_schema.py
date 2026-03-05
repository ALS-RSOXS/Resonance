from __future__ import annotations

import sqlite3

import pytest

from resonance.api.data.schema import (
    BEAMTIME_SCHEMA_VERSION,
    create_beamtime_schema,
    create_index_schema,
    migrate_beamtime_schema,
)


def test_create_beamtime_schema_tables() -> None:
    conn = sqlite3.connect(":memory:")
    create_beamtime_schema(conn)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"samples", "runs", "streams", "events", "image_refs"}.issubset(tables)
    conn.close()


def test_create_beamtime_schema_foreign_keys() -> None:
    conn = sqlite3.connect(":memory:")
    create_beamtime_schema(conn)
    conn.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO runs (uid, sample_id, plan_name, time_start) VALUES (?, ?, ?, ?)",
            ("uid-1", 9999, "nexafs", 1.0),
        )
        conn.commit()
    conn.close()


def test_create_beamtime_schema_idempotent() -> None:
    conn = sqlite3.connect(":memory:")
    create_beamtime_schema(conn)
    create_beamtime_schema(conn)
    conn.close()


def test_create_beamtime_schema_user_version() -> None:
    conn = sqlite3.connect(":memory:")
    create_beamtime_schema(conn)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == BEAMTIME_SCHEMA_VERSION
    conn.close()


def test_migrate_beamtime_schema_already_current() -> None:
    conn = sqlite3.connect(":memory:")
    create_beamtime_schema(conn)
    migrate_beamtime_schema(conn, target_version=BEAMTIME_SCHEMA_VERSION)
    conn.close()


def test_migrate_beamtime_schema_downgrade_raises() -> None:
    conn = sqlite3.connect(":memory:")
    create_beamtime_schema(conn)
    with pytest.raises(RuntimeError):
        migrate_beamtime_schema(conn, target_version=0)
    conn.close()


def test_migrate_beamtime_schema_upgrade_raises_not_implemented() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    with pytest.raises(NotImplementedError):
        migrate_beamtime_schema(conn, target_version=1)
    conn.close()


def test_create_index_schema_tables() -> None:
    conn = sqlite3.connect(":memory:")
    create_index_schema(conn)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"researchers", "beamtimes", "runs_index"}.issubset(tables)
    conn.close()


def test_events_seq_num_unique_constraint() -> None:
    conn = sqlite3.connect(":memory:")
    create_beamtime_schema(conn)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO streams (uid, run_uid, name, data_keys, time_created) VALUES (?, ?, ?, ?, ?)",
        ("s1", "run-1", "primary", "{}", 1.0),
    )
    conn.commit()
    conn.execute(
        "INSERT INTO events (uid, stream_uid, seq_num, time, data) VALUES (?, ?, ?, ?, ?)",
        ("e1", "s1", 1, 1.0, "{}"),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO events (uid, stream_uid, seq_num, time, data) VALUES (?, ?, ?, ?, ?)",
            ("e2", "s1", 1, 2.0, "{}"),
        )
        conn.commit()
    conn.close()


def test_streams_name_unique_per_run() -> None:
    conn = sqlite3.connect(":memory:")
    create_beamtime_schema(conn)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO runs (uid, plan_name, time_start) VALUES (?, ?, ?)",
        ("run-1", "nexafs", 1.0),
    )
    conn.commit()
    conn.execute(
        "INSERT INTO streams (uid, run_uid, name, data_keys, time_created) VALUES (?, ?, ?, ?, ?)",
        ("s1", "run-1", "primary", "{}", 1.0),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO streams (uid, run_uid, name, data_keys, time_created) VALUES (?, ?, ?, ?, ?)",
            ("s2", "run-1", "primary", "{}", 2.0),
        )
        conn.commit()
    conn.close()
