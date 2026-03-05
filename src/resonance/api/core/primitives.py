"""Low-level safe async primitives for beamline control"""

import asyncio
import time
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from bcs.BCSz import BCSServer, MotorStatus

from resonance.api.types import MotorError, MotorTimeoutError, ScanAbortedError


class AbortFlag:
    """Thread-safe abort flag for scan cancellation in Jupyter notebooks"""

    def __init__(self):
        self._aborted = False
        self._lock = asyncio.Lock()

    async def set(self):
        """Set the abort flag"""
        async with self._lock:
            self._aborted = True

    async def is_set(self) -> bool:
        """Check if abort flag is set"""
        async with self._lock:
            return self._aborted

    async def clear(self):
        """Clear the abort flag"""
        async with self._lock:
            self._aborted = False


async def wait_for_motors(
    server: BCSServer,
    motors: list[str],
    timeout: float = 30.0,
    check_interval: float = 0.05,
    abort_flag: AbortFlag | None = None,
) -> None:
    """
    Wait for all motors to complete movement.

    Parameters
    ----------
    server : BCSServer
        BCS server instance
    motors : list[str]
        Motor names to wait for
    timeout : float
        Maximum time to wait in seconds
    check_interval : float
        Time between status checks
    abort_flag : AbortFlag or None
        Optional abort flag to check

    Raises
    ------
    MotorTimeoutError
        If timeout exceeded
    ScanAbortedError
        If abort_flag is set
    """
    start_time = time.time()

    while True:
        if abort_flag and await abort_flag.is_set():
            raise ScanAbortedError("Scan aborted by user")

        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise MotorTimeoutError(
                f"Motors {motors} did not complete within {timeout}s"
            )

        response = await server.get_motor(motors=motors)
        all_complete = True

        for motor_data in response["data"]:
            status = MotorStatus(motor_data["status"])
            if not status.is_set(MotorStatus.MOVE_COMPLETE):
                all_complete = False
                break

        if all_complete:
            return

        await asyncio.sleep(check_interval)


async def wait_for_settle(delay: float, abort_flag: AbortFlag | None = None) -> None:
    """
    Wait for motor settling with abort check.

    Parameters
    ----------
    delay : float
        Time to wait in seconds
    abort_flag : AbortFlag or None
        Optional abort flag to check

    Raises
    ------
    ScanAbortedError
        If abort_flag is set during wait
    """
    if delay <= 0:
        return

    steps = int(delay / 0.1)
    for _ in range(steps):
        if abort_flag and await abort_flag.is_set():
            raise ScanAbortedError("Scan aborted during settle")
        await asyncio.sleep(0.1)

    remainder = delay - (steps * 0.1)
    if remainder > 0:
        if abort_flag and await abort_flag.is_set():
            raise ScanAbortedError("Scan aborted during settle")
        await asyncio.sleep(remainder)


@asynccontextmanager
async def motor_move(
    server: BCSServer,
    motors: dict[str, float],
    timeout: float = 30.0,
    backlash: bool = True,
    restore_on_exit: bool = True,
) -> AsyncGenerator[dict[str, float]]:
    """
    Context manager for safe motor movements with automatic position restoration.

    Parameters
    ----------
    server : BCSServer
        BCS server instance
    motors : dict[str, float]
        Motor name to target position mapping
    timeout : float
        Timeout for motor moves
    backlash : bool
        Use backlash compensation
    restore_on_exit : bool
        Restore initial positions on exit

    Yields
    ------
    dict[str, float]
        Initial motor positions

    Raises
    ------
    MotorError
        If motor move fails

    Examples
    --------
    >>> async with motor_move(server, {"Sample X": 10.0}) as initial_pos:
    ...     pass
    """
    command = "Backlash Move" if backlash else "Normal Move"
    initial_response = await server.get_motor(motors=list(motors.keys()))
    initial_pos = {m["motor"]: m["position"] for m in initial_response["data"]}

    try:
        await server.command_motor(
            commands=[command] * len(motors),
            motors=list(motors.keys()),
            goals=list(motors.values()),
        )

        await wait_for_motors(server, list(motors.keys()), timeout=timeout)

        yield initial_pos

    except Exception as e:
        raise MotorError(f"Motor move failed: {e}") from e

    finally:
        if restore_on_exit:
            try:
                restore_command = "Backlash Move" if backlash else "Normal Move"
                await server.command_motor(
                    commands=[restore_command] * len(initial_pos),
                    motors=list(initial_pos.keys()),
                    goals=list(initial_pos.values()),
                )
                await wait_for_motors(server, list(initial_pos.keys()), timeout=timeout)
            except Exception as e:
                warnings.warn(f"Failed to restore motor positions: {e}", stacklevel=2)


@asynccontextmanager
async def shutter_control(
    server: BCSServer, shutter: str = "Light Output", delay_before_open: float = 0.0
) -> AsyncGenerator[None]:
    """
    Context manager for safe shutter control.
    Guarantees shutter closes even on exception.

    Parameters
    ----------
    server : BCSServer
        BCS server instance
    shutter : str
        Shutter DIO channel name
    delay_before_open : float
        Delay before opening shutter

    Examples
    --------
    >>> async with shutter_control(server):
    ...     await server.acquire_data(time=1.0)
    """
    try:
        if delay_before_open > 0:
            await asyncio.sleep(delay_before_open)

        await server.set_do(chan=shutter, value=True)
        yield

    finally:
        try:
            await server.set_do(chan=shutter, value=False)
        except Exception as e:
            warnings.warn(f"Failed to close shutter: {e}", stacklevel=2)
