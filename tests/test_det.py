from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from resonance.api.core.det import (
    DETECTOR_NAME,
    AreaDetector,
    ExposureQuality,
    _parse_acquired2d_string,
)


def test_parse_acquired2d_string_basic() -> None:
    payload = {"Height": 2, "Width": 3, "Data": "1,2,3,4,5,6,"}
    result = _parse_acquired2d_string(payload)
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


def test_parse_acquired2d_string_shape() -> None:
    values = ",".join(str(i) for i in range(16))
    payload = {"Height": 4, "Width": 4, "Data": values}
    result = _parse_acquired2d_string(payload)
    assert result.shape == (4, 4)


def test_acquire_success() -> None:
    conn = MagicMock()
    conn.start_instrument_acquire = AsyncMock(return_value={"success": True})
    conn.get_instrument_acquired2d_string = AsyncMock(
        return_value={"Height": 2, "Width": 2, "Data": "10,20,30,40"}
    )
    det = AreaDetector(conn)
    result = asyncio.run(det.acquire(0.5))
    expected = np.array([[10, 20], [30, 40]], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)
    assert det._last_shape == (2, 2)


def test_acquire_failure_returns_none() -> None:
    conn = MagicMock()
    conn.start_instrument_acquire = AsyncMock(
        return_value={"success": False, "error description": "timeout"}
    )
    det = AreaDetector(conn)
    result = asyncio.run(det.acquire(0.5))
    assert result is None
    assert det._last_shape is None


def test_check_exposure_underexposed() -> None:
    conn = MagicMock()
    det = AreaDetector(conn)
    image = np.zeros((1000, 1000), dtype=np.int32)
    quality = det.check_exposure(image)
    assert quality.underexposed is True
    assert quality.overexposed is False


def test_check_exposure_overexposed() -> None:
    conn = MagicMock()
    det = AreaDetector(conn)
    image = np.full((100, 100), 250_000, dtype=np.int32)
    quality = det.check_exposure(image)
    assert quality.overexposed is True


def test_check_exposure_suggested_is_none() -> None:
    conn = MagicMock()
    det = AreaDetector(conn)
    image = np.zeros((10, 10), dtype=np.int32)
    quality = det.check_exposure(image)
    assert quality.suggested_exposure_seconds is None


def test_describe_before_acquire() -> None:
    conn = MagicMock()
    det = AreaDetector(conn)
    desc = det.describe()
    assert desc["shape"] == []


def test_describe_after_acquire() -> None:
    conn = MagicMock()
    conn.start_instrument_acquire = AsyncMock(return_value={"success": True})
    conn.get_instrument_acquired2d_string = AsyncMock(
        return_value={"Height": 2, "Width": 2, "Data": "10,20,30,40"}
    )
    det = AreaDetector(conn)
    asyncio.run(det.acquire(0.5))
    assert det.describe()["shape"] == [2, 2]


def test_detector_name_constant() -> None:
    assert DETECTOR_NAME == "Axis Photonique"


def test_exposure_quality_dataclass_fields() -> None:
    q = ExposureQuality(overexposed=True, underexposed=False, suggested_exposure_seconds=0.5)
    assert q.overexposed is True
    assert q.underexposed is False
    assert q.suggested_exposure_seconds == 0.5
