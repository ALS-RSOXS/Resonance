from __future__ import annotations

from typing import TYPE_CHECKING, get_args

from resonance.api.types import DIO, ShutterError

if TYPE_CHECKING:
    from bcs import BCSz

_valid_channels: tuple[str, ...] = get_args(DIO.__value__)

__all__ = ["DIOAccessor"]


class DIOAccessor:
    """
    Async interface for reading and writing BCS digital I/O channels.

    Parameters
    ----------
    conn : BCSz.BCSServer
        Active BCSz server connection.

    Notes
    -----
    Wraps BCSz digital I/O channels. `read` reads digital inputs; `set` writes
    digital outputs. The shutter channel ("Light Output", "Shutter Output", etc.)
    is the most common use case. Values are always coerced to bool.

    Examples
    --------
    >>> state = await bl.dio.read("Shutter Output", "Light Output")
    >>> print(state)
    {'Shutter Output': True, 'Light Output': True}
    >>> await bl.dio.set("Shutter Output", False)
    >>> await bl.dio.set("Light Output", True)
    >>> state = await bl.dio.read("Shutter Output", "Light Output")
    >>> print(state)
    {'Shutter Output': False, 'Light Output': True}
    """

    def __init__(self, conn: BCSz.BCSServer) -> None:
        self._conn = conn

    async def read(self, *channels: str) -> dict[str, bool]:
        """
        Read current state of one or more digital I/O channels.

        Parameters
        ----------
        *channels : str
            One or more DIO channel names to read. Must be members of the
            ``DIO`` literal type.

        Returns
        -------
        dict[str, bool]
            Mapping of channel name to its current boolean state.

        Raises
        ------
        KeyError
            If any channel name is not a valid DIO channel.

        Notes
        -----
        Channels are read in a single BCSz ``get_di`` call. The response
        ``data`` field is coerced to bool for each channel.
        """
        invalid = [c for c in channels if c not in _valid_channels]
        if invalid:
            raise KeyError(
                f"Unknown DIO channel(s): {invalid}. Valid channels: {list(_valid_channels)}"
            )

        response: dict = await self._conn.get_di(chans=list(channels))
        return {
            chan: bool(data)
            for chan, data in zip(response["chans"], response["data"])
        }

    async def set(self, channel: str, value: bool | int) -> None:
        """
        Set a digital output channel to True (on) or False (off).

        Parameters
        ----------
        channel : str
            DIO channel name to write. Must be a member of the ``DIO`` literal type.
        value : bool or int
            Output value to set. Non-zero integers map to True, zero maps to False.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If ``channel`` is not a valid DIO channel name.
        ValueError
            If ``value`` is not a bool or int (e.g. a float is passed).
        ShutterError
            If the BCSz ``set_do`` call raises an exception.

        Notes
        -----
        Value is coerced to bool: any non-zero int is True, 0 is False.
        Typical use is controlling the beamline shutter (e.g. channel="Light Output").
        """
        if channel not in _valid_channels:
            raise KeyError(
                f"Unknown DIO channel: {channel!r}. Valid channels: {list(_valid_channels)}"
            )

        if not isinstance(value, bool | int):
            raise ValueError(
                f"value must be bool or int, got {type(value).__name__!r}"
            )

        bool_value = bool(value)

        try:
            await self._conn.set_do(chan=channel, value=bool_value)
        except Exception as exc:
            raise ShutterError(
                f"Failed to set DIO channel {channel!r} to {bool_value}: {exc}"
            ) from exc

    # TODO: add set_many for batch digital output (e.g. enable multiple outputs atomically)
