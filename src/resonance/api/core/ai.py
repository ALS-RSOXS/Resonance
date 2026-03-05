from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np
from uncertainties import Variable, ufloat

from resonance.api.types import AI, AcquisitionError

if TYPE_CHECKING:
    from bcs import BCSz

_AI_CHANNELS: tuple[str, ...] = get_args(AI.__value__)


class AIAccessor:
    """Accessor for analog input channels via the BCS server.

    Parameters
    ----------
    conn : BCSz.BCSServer
        Active connection to the BCS hardware server.

    Notes
    -----
    Wraps `acquire_data` and `get_acquired_array` for typed, validated
    access to AI channels defined in `resonance.api.types.AI`.
    """

    def __init__(self, conn: BCSz.BCSServer) -> None:
        self._conn = conn

    async def read(self, *channels: str) -> dict[str, list[float]]:
        """Return the last-acquired raw array for each channel.

        Parameters
        ----------
        *channels : str
            One or more AI channel names from `resonance.api.types.AI`.

        Returns
        -------
        dict[str, list[float]]
            Mapping of channel name to raw sample array.

        Raises
        ------
        KeyError
            If any channel name is not a valid AI channel.
        AcquisitionError
            If the BCS response contains an empty data array for a channel.

        Notes
        -----
        Does not trigger a new acquisition. Returns the most recent data
        buffered by BCS. Call `trigger_and_read` to acquire fresh data.
        Use `read` only when data was already acquired (e.g. after `acquire_data`
        was called externally).
        """
        invalid = [c for c in channels if c not in _AI_CHANNELS]
        if invalid:
            raise KeyError(
                f"Invalid AI channel(s): {invalid}. Valid channels: {list(_AI_CHANNELS)}"
            )

        response: dict = await self._conn.get_acquired_array(chans=list(channels))
        result: dict[str, list[float]] = {}
        for entry in response["chans"]:
            name: str = entry["chan"]
            data: list[float] = entry["data"]
            if not data:
                raise AcquisitionError(f"Empty data returned for channel '{name}'")
            result[name] = data
        return result

    async def trigger_and_read(
        self,
        channels: list[str],
        acquisition_time: float = 1.0,
    ) -> dict[str, Variable]:
        """Trigger acquisition and return mean and standard error per channel.

        Parameters
        ----------
        channels : list[str]
            AI channel names from `resonance.api.types.AI`.
        acquisition_time : float, optional
            Duration of acquisition in seconds. Must be positive. Default is 1.0.

        Returns
        -------
        dict[str, Variable]
            Mapping of channel name to `ufloat(mean, std_err)`.

        Raises
        ------
        KeyError
            If any channel name is not a valid AI channel.
        ValueError
            If `acquisition_time` is not strictly positive.
        AcquisitionError
            If the BCS response contains an empty data array for a channel.

        Notes
        -----
        Performs a blocking acquisition of `acquisition_time` seconds.
        Use per scan point to get mean and standard error for each channel.
        Standard error is computed as std_dev / sqrt(N). For N=1, std_err is 0.

        Examples
        --------
        >>> data = await bl.ai.trigger_and_read(["Photodiode", "TEY signal"], acquisition_time=1.0)
        >>> print(data["Photodiode"])  # ufloat(mean, std_err)
        """
        invalid = [c for c in channels if c not in _AI_CHANNELS]
        if invalid:
            raise KeyError(
                f"Invalid AI channel(s): {invalid}. Valid channels: {list(_AI_CHANNELS)}"
            )
        if acquisition_time <= 0:
            raise ValueError(
                f"acquisition_time must be positive, got {acquisition_time}"
            )

        await self._conn.acquire_data(chans=channels, time=acquisition_time)  # pyright: ignore[reportArgumentType]
        response: dict = await self._conn.get_acquired_array(chans=channels)

        # TODO: add optional return of raw arrays for debugging or downstream processing
        result: dict[str, Variable] = {}
        for entry in response["chans"]:
            name: str = entry["chan"]
            data: list[float] = entry["data"]
            if not data:
                raise AcquisitionError(f"Empty data returned for channel '{name}'")
            arr = np.asarray(data, dtype=float)
            mean = float(np.nanmean(arr))
            std_err = (
                float(np.nanstd(arr, ddof=1) / np.sqrt(len(arr)))
                if len(arr) > 1
                else 0.0
            )
            result[name] = ufloat(mean, std_err)
        return result
