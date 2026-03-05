"""MCP tools for motor positions and status."""

from typing import Any

from resonance.api.types import motor

from ..connection import connection_manager
from ..models import MotorListResponse, MotorPositionsResponse, MotorStatusResponse


async def list_motors() -> dict[str, Any]:
    """
    List all available motor names.

    Returns
    -------
    dict
        Response with motors list
    """
    return MotorListResponse(motors=list(motor)).model_dump()


async def get_motor_positions(motors: list[str] | None = None) -> dict[str, Any]:
    """
    Get current positions for motors.

    Parameters
    ----------
    motors : list[str] | None, optional
        List of motor names to get positions for. If None or empty, gets all motors.

    Returns
    -------
    dict
        Response with motor positions

    Raises
    ------
    RuntimeError
        If server communication fails
    ValueError
        If invalid motor names are provided
    """
    try:
        await connection_manager.ensure_connected()
        beamline = await connection_manager.get_server()

        target = motors if motors else list(motor)
        result = await beamline.motors.read(*target)
        positions = {name: state.position for name, state in result.items()}
        return MotorPositionsResponse(positions=positions).model_dump()

    except (ConnectionError, RuntimeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error getting motor positions: {e}") from e


async def get_motor_status(motors: list[str] | None = None) -> dict[str, Any]:
    """
    Get full status for motors including position, goal, and status bits.

    Parameters
    ----------
    motors : list[str] | None, optional
        List of motor names to get status for. If None or empty, gets all motors.

    Returns
    -------
    dict
        Response with motor status information

    Raises
    ------
    RuntimeError
        If server communication fails
    ValueError
        If invalid motor names are provided
    """
    try:
        await connection_manager.ensure_connected()
        beamline = await connection_manager.get_server()

        target = motors if motors else list(motor)
        result = await beamline.motors.read(*target)
        status_dict = {
            name: {
                "position": state.position,
                "goal": state.goal,
                "status": state.status,
            }
            for name, state in result.items()
        }
        return MotorStatusResponse(status=status_dict).model_dump()

    except (ConnectionError, RuntimeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error getting motor status: {e}") from e
