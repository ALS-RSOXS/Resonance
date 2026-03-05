"""MCP tools for digital I/O (DIO) channels."""

from typing import Any

from resonance.api.types import dio

from ..connection import connection_manager
from ..models import DIOChannelsResponse, DIOStatesResponse


async def list_dio_channels() -> dict[str, Any]:
    """
    List all available DIO channel names.

    Returns
    -------
    dict
        Response with channels list
    """
    return DIOChannelsResponse(channels=list(dio)).model_dump()


async def get_dio_states(channels: list[str] | None = None) -> dict[str, Any]:
    """
    Get current states for DIO channels.

    Parameters
    ----------
    channels : list[str] | None, optional
        List of channel names to get states for. If None or empty, gets all channels.

    Returns
    -------
    dict
        Response with channel states (boolean values)

    Raises
    ------
    RuntimeError
        If server communication fails
    ValueError
        If invalid channel names are provided
    """
    try:
        await connection_manager.ensure_connected()
        beamline = await connection_manager.get_server()

        target = channels if channels else list(dio)
        result = await beamline.dio.read(*target)
        states = {chan: bool(val) for chan, val in result.items()}
        return DIOStatesResponse(states=states).model_dump()

    except (ConnectionError, RuntimeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error getting DIO states: {e}") from e
