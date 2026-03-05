from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

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


def test_write_image_creates_zarr_and_image_refs(tmp_path: Path) -> None:
    db = tmp_path / "bt.db"
    image = np.ones((4, 4), dtype=np.int32)
    with RunWriter(db, SampleMetadata(name="PS")) as w:
        w.open_run("scan")
        w.open_stream("primary", {})
        event_uid = w.write_event({"Energy": 285.0})
        w.write_image(event_uid, "detector_image", image)

    db_conn = sqlite3.connect(db)
    row = db_conn.execute(
        "SELECT shape_x, shape_y, dtype, compression_codec, zarr_group FROM image_refs"
    ).fetchone()
    db_conn.close()

    assert row is not None
    shape_x, shape_y, dtype, codec, zarr_group = row
    assert shape_x == 4
    assert shape_y == 4
    assert dtype == "int32"
    assert codec == "blosc"

    store = zarr.open_group(str(tmp_path / "bt.zarr"), mode="r")
    assert store[zarr_group].shape == (1, 4, 4)


def test_write_image_raises_without_open_stream(tmp_path: Path) -> None:
    db = tmp_path / "bt.db"
    w = RunWriter(db, SampleMetadata(name="PS"))
    w.open()
    w.open_run("scan")
    with pytest.raises(RuntimeError):
        w.write_image("x", "field", np.ones((4, 4), dtype=np.int32))
    w.close_run()
    w.close()


def test_write_image_multiple_frames(tmp_path: Path) -> None:
    db = tmp_path / "bt.db"
    with RunWriter(db, SampleMetadata(name="PS")) as w:
        w.open_run("scan")
        w.open_stream("primary", {})
        for i in range(3):
            event_uid = w.write_event({"i": float(i)})
            w.write_image(event_uid, "detector_image", np.ones((4, 4), dtype=np.int32))

    db_conn = sqlite3.connect(db)
    rows = db_conn.execute(
        "SELECT index_in_stack, zarr_group FROM image_refs ORDER BY index_in_stack"
    ).fetchall()
    db_conn.close()

    assert len(rows) == 3
    assert [r[0] for r in rows] == [0, 1, 2]

    store = zarr.open_group(str(tmp_path / "bt.zarr"), mode="r")
    assert store[rows[0][1]].shape == (3, 4, 4)
