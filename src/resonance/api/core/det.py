from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from bcs import BCSz

DETECTOR_NAME: Final[str] = "Axis Photonique"


async def get_acquired2d_string(conn: BCSz.BCSServer, name: str) -> dict[str, Any]:
    """
    Get the acquired 2D string from the detector.
    """
    return await conn.bcs_request("GetInstrumentAcquired2DString", dict(locals()))


@dataclass
class ExposureQuality:
    """
    Quality assessment of a single detector exposure.

    Attributes
    ----------
    overexposed : bool
        True when the number of saturated pixels exceeds the configured threshold.
    underexposed : bool
        True when the number of dark pixels exceeds the configured threshold.
    suggested_exposure_seconds : float or None
        Recommended exposure time, or None when no suggestion is available.
    """

    overexposed: bool
    underexposed: bool
    suggested_exposure_seconds: float | None


def _parse_acquired2d_string(payload: dict[str, Any]) -> np.ndarray:
    """
    Parse the BCSz GetInstrumentAcquired2DString response into a 2-D array.

    Parameters
    ----------
    payload : dict[str, Any]
        Response dict containing ``"Height"``, ``"Width"``, and ``"Data"`` keys.
        ``"Data"`` is a comma-separated string of integer pixel values.

    Returns
    -------
    np.ndarray
        Integer array of dtype ``int32`` and shape ``(height, width)``.
    """
    height: int = int(payload["Height"])
    width: int = int(payload["Width"])
    tokens = [t for t in payload["Data"].split(",") if t.strip()]
    return np.array(tokens, dtype=np.int32).reshape(height, width)


class AreaDetector:
    """
    Interface to the Axis Photonique 2-D area detector via BCSz.

    Parameters
    ----------
    conn : BCSz.BCSServer
        Active BCSz server connection.
    name : str, optional
        Instrument name registered in BCSz, defaults to ``DETECTOR_NAME``.
    """

    def __init__(self, conn: BCSz.BCSServer, *, name: str = DETECTOR_NAME) -> None:
        self._conn = conn
        self._name = name
        self._last_shape: tuple[int, int] | None = None

    async def acquire(self, exposure_seconds: float) -> np.ndarray | None:
        """
        Trigger an exposure and return the acquired image.

        Parameters
        ----------
        exposure_seconds : float
            Integration time in seconds.

        Returns
        -------
        np.ndarray or None
            2-D ``int32`` array of shape ``(height, width)``, or ``None`` if
            the acquisition reported failure.

        Notes
        -----
        The shutter is detector-driven; no plan-level shutter wrapping is
        required around this call.
        """
        res = await self._conn.start_instrument_acquire(
            name=self._name,
            run_type="Exposure",
            acq_time_s=exposure_seconds,  # pyright: ignore[reportArgumentType]
        )
        if not res.get("success"):
            return None
        raw = await get_acquired2d_string(self._conn, self._name)
        image = _parse_acquired2d_string(raw)
        self._last_shape = (image.shape[0], image.shape[1])
        return image

    def check_exposure(
        self,
        image: np.ndarray,
        *,
        over_threshold: int = int(2e16),
        over_pixel_count: int = 500,
        under_threshold: int = 50,
        under_pixel_count: int = 950_000,
    ) -> ExposureQuality:
        """
        Assess whether an image is over- or under-exposed.

        Parameters
        ----------
        image : np.ndarray
            2-D detector image.
        over_threshold : int, optional
            Pixel value above which a pixel is considered saturated.
        over_pixel_count : int, optional
            Minimum number of saturated pixels required to flag overexposure.
        under_threshold : int, optional
            Pixel value below which a pixel is considered dark.
        under_pixel_count : int, optional
            Minimum number of dark pixels required to flag underexposure.

        Returns
        -------
        ExposureQuality
            Dataclass with ``overexposed``, ``underexposed``, and
            ``suggested_exposure_seconds`` fields.

        Notes
        -----
        The pixel-count heuristic mirrors the sst-rsoxs GreatEyes thresholds
        used for automated exposure quality decisions.
        """
        overexposed = int(np.sum(image >= over_threshold)) >= over_pixel_count
        underexposed = int(np.sum(image < under_threshold)) >= under_pixel_count
        return ExposureQuality(
            overexposed=overexposed,
            underexposed=underexposed,
            suggested_exposure_seconds=None,
        )

    def describe(self) -> dict[str, Any]:
        """
        Return a data_keys-compatible descriptor dict for use in RunWriter.open_stream.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``dtype``, ``source``, ``external``, and
            ``shape``.  ``shape`` reflects the last successfully acquired
            image dimensions, or an empty list when no image has been
            acquired yet.
        """
        return {
            "dtype": "int32",
            "source": "detector",
            "external": True,
            "shape": list(self._last_shape) if self._last_shape is not None else [],
        }
