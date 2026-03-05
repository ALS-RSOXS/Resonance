from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from resonance.api.data.models import SampleMetadata
from resonance.api.data.writer import RunWriter


def test_run_writer_context_manager(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with RunWriter(db, SampleMetadata(name="PS")) as w:
        w.open_run("nexafs")
        w.open_stream("primary", {"Energy": {}})
        w.write_event({"Energy": 285.0}, {"Energy": 1234567890.0})
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM streams").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    conn.close()


def test_run_writer_multiple_events(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with RunWriter(db, SampleMetadata(name="PS")) as w:
        w.open_run("nexafs")
        w.open_stream("primary", {})
        for _ in range(5):
            w.write_event({})
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT seq_num FROM events ORDER BY seq_num").fetchall()
    conn.close()
    assert len(rows) == 5
    assert [r[0] for r in rows] == [1, 2, 3, 4, 5]


def test_run_writer_sample_upsert(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with RunWriter(db, SampleMetadata(name="PS")) as w:
        w.open_run("nexafs")
    with RunWriter(db, SampleMetadata(name="PS")) as w2:
        w2.open_run("nexafs")
    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    conn.close()
    assert count == 1


def test_run_writer_exit_status_aborted(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    w = RunWriter(db, SampleMetadata(name="PS"))
    w.open()
    w.open_run("nexafs")
    w.close_run(exit_status="aborted")
    w.close()
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT exit_status FROM runs").fetchone()
    conn.close()
    assert row[0] == "aborted"


def test_run_writer_exception_sets_failed_status(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with pytest.raises(ValueError), RunWriter(db, SampleMetadata(name="PS")) as w:
        w.open_run("nexafs")
        w.open_stream("primary", {})
        raise ValueError("abort")
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT exit_status FROM runs").fetchone()
    conn.close()
    assert row[0] == "failed"


def test_write_event_raises_without_open_run(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    w = RunWriter(db, SampleMetadata(name="PS"))
    w.open()
    with pytest.raises(RuntimeError):
        w.write_event({})
    w.close()


def test_run_writer_close_run_clears_uid(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    w = RunWriter(db, SampleMetadata(name="PS"))
    w.open()
    w.open_run("nexafs")
    w.open_stream("primary", {})
    w.close_run()
    with pytest.raises(RuntimeError):
        w.write_event({})
    w.close()
