"""MCP tools for analog input (AI) channels."""

from typing import Any

from resonance.api.types import ai

from ..connection import connection_manager
from ..models import AIChannelResponse, AIUncertaintyResponse, AIValuesResponse


async def list_ai_channels() -> dict[str, Any]:
    """
    List all available AI channel names.

    Returns
    -------
    dict
        Response with channels list
    """
    return AIChannelResponse(channels=list(ai)).model_dump()


async def get_ai_values(channels: list[str] | None = None) -> dict[str, Any]:
    """
    Get current values for AI channels.

    Parameters
    ----------
    channels : list[str] | None, optional
        List of channel names to get values for. If None or empty, gets all channels.

    Returns
    -------
    dict
        Response with channel values

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

        target = channels if channels else list(ai)
        result = await beamline.ai.read(*target)
        values = {
            chan: float(vals[-1]) if vals else 0.0 for chan, vals in result.items()
        }
        return AIValuesResponse(values=values).model_dump()

    except (ConnectionError, RuntimeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error getting AI values: {e}") from e


async def get_ai_with_uncertainty(
    channels: list[str],
    acquisition_time: float = 1.0,
) -> dict[str, Any]:
    """
    Get AI channel values with uncertainty (mean and standard deviation).

    Parameters
    ----------
    channels : list[str]
        List of channel names to acquire
    acquisition_time : float, optional
        Acquisition time in seconds (default: 1.0)

    Returns
    -------
    dict
        Response with channel values including mean and std

    Raises
    ------
    RuntimeError
        If server communication fails
    ValueError
        If channels is empty or acquisition_time is non-positive
    """
    try:
        if not channels:
            raise ValueError("channels list cannot be empty")
        if acquisition_time <= 0:
            raise ValueError("acquisition_time must be positive")

        await connection_manager.ensure_connected()
        beamline = await connection_manager.get_server()

        ufloat_data = await beamline.ai.trigger_and_read(channels, acquisition_time)
        values = {
            chan: {"mean": float(uval.nominal_value), "std": float(uval.std_dev)}
            for chan, uval in ufloat_data.items()
        }
        return AIUncertaintyResponse(values=values).model_dump()

    except (ConnectionError, RuntimeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error getting AI values with uncertainty: {e}") from e
