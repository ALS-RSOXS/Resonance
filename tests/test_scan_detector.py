from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import zarr

from resonance.api.core.det import AreaDetector
from resonance.api.core.scan import ScanExecutor, ScanPlan
from resonance.api.data.models import SampleMetadata
from resonance.api.data.writer import RunWriter

if TYPE_CHECKING:
    from pathlib import Path


def _make_scan_plan() -> ScanPlan:
    df = pd.DataFrame(
        {
            "Sample X": [0.0, 1.0],
            "exposure": [0.002, 0.002],
        }
    )
    return ScanPlan.from_dataframe(df, ai_channels=["Photodiode"])


@asynccontextmanager
async def _noop_motor_move(*args, **kwargs):
    yield {}


@asynccontextmanager
async def _noop_shutter(*args, **kwargs):
    yield


def _make_conn() -> MagicMock:
    conn = MagicMock()
    conn.acquire_data = AsyncMock(return_value=None)
    conn.get_acquired_array = AsyncMock(
        return_value={"chans": [{"chan": "Photodiode", "data": [1.0]}]}
    )
    return conn


def test_execute_scan_with_detector_writes_images(tmp_path: Path) -> None:
    db_path = tmp_path / "bt.db"
    conn = _make_conn()
    det = MagicMock(spec=AreaDetector)
    det.acquire = AsyncMock(return_value=np.ones((4, 4), dtype=np.int32))
    det.describe = MagicMock(
        return_value={
            "dtype": "int32",
            "source": "detector",
            "external": True,
            "shape": [4, 4],
        }
    )
    writer = RunWriter(db_path, SampleMetadata(name="PS"))
    writer.open()
    executor = ScanExecutor(conn)
    scan_plan = _make_scan_plan()

    with (
        patch("resonance.api.core.scan.motor_move", _noop_motor_move),
        patch("resonance.api.core.scan.shutter_control", _noop_shutter),
        patch("resonance.api.core.scan.wait_for_settle", AsyncMock()),
    ):
        asyncio.run(
            executor.execute_scan(scan_plan, progress=False, writer=writer, detector=det)
        )

    writer.close()

    db_conn = sqlite3.connect(db_path)
    rows = db_conn.execute(
        "SELECT shape_x, shape_y, dtype, zarr_group FROM image_refs"
    ).fetchall()
    db_conn.close()

    assert len(rows) == 2
    for row in rows:
        assert row[0] == 4
        assert row[1] == 4
        assert row[2] == "int32"
        assert row[3].startswith("runs/")

    zarr_group = rows[0][3]
    store = zarr.open_group(str(tmp_path / "bt.zarr"), mode="r")
    arr = store[zarr_group]
    assert arr.shape == (2, 4, 4)


def test_execute_scan_without_detector_no_images(tmp_path: Path) -> None:
    db_path = tmp_path / "bt.db"
    conn = _make_conn()
    writer = RunWriter(db_path, SampleMetadata(name="PS"))
    writer.open()
    executor = ScanExecutor(conn)
    scan_plan = _make_scan_plan()

    with (
        patch("resonance.api.core.scan.motor_move", _noop_motor_move),
        patch("resonance.api.core.scan.shutter_control", _noop_shutter),
        patch("resonance.api.core.scan.wait_for_settle", AsyncMock()),
    ):
        asyncio.run(
            executor.execute_scan(scan_plan, progress=False, writer=writer, detector=None)
        )

    writer.close()

    db_conn = sqlite3.connect(db_path)
    count = db_conn.execute("SELECT COUNT(*) FROM image_refs").fetchone()[0]
    db_conn.close()
    assert count == 0
