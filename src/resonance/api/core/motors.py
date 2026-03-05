from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args

from resonance.api.core.primitives import AbortFlag, wait_for_motors
from resonance.api.types import (
    Command,
    Motor,
    MotorError,
    MotorTimeoutError,
    ScanAbortedError,
)

if TYPE_CHECKING:
    from bcs import BCSz

__all__ = [
    "MotorState",
    "MotorAccessor",
    "MotorError",
    "MotorTimeoutError",
    "ScanAbortedError",
]

_valid_motors: tuple[str, ...] = get_args(Motor.__value__)
_valid_commands: tuple[str, ...] = get_args(Command.__value__)


@dataclass
class MotorState:
    """
    Snapshot of a single motor's state from BCS.

    Parameters
    ----------
    position : float
        Current encoder position reported by BCS.
    goal : float
        Commanded target position.
    status : int
        Raw BCS status code for the motor.
    time : float
        Timestamp of the reading as reported by BCS.
    """

    position: float
    goal: float
    status: int  # TODO: replace with a MotorStatus enum once BCSz MotorStatus values are documented
    time: float


class MotorAccessor:
    """
    High-level async interface for reading and commanding BCS motors.

    Wraps the BCSz.BCSServer motor API with name validation, typed return
    values, and cooperative cancellation support.

    Parameters
    ----------
    conn : BCSz.BCSServer
        Active BCS server connection.

    Examples
    --------
    >>> motors = MotorAccessor(server)
    >>> state = await motors.read("Sample X", "Sample Y")
    >>> await motors.set("Sample X", 10.0)
    >>> await motors.wait(["Sample X"])
    """

    def __init__(self, conn: BCSz.BCSServer) -> None:
        self._conn = conn

    def _validate_motors(self, names: tuple[str, ...] | list[str]) -> None:
        invalid = [n for n in names if n not in _valid_motors]
        if invalid:
            raise KeyError(
                f"Unknown motor name(s): {invalid}. "
                f"Valid motors are defined in resonance.api.types.Motor."
            )

    def _validate_command(self, cmd: str) -> None:
        if cmd not in _valid_commands:
            raise ValueError(
                f"Unknown command: {cmd!r}. "
                f"Valid commands are defined in resonance.api.types.Command."
            )

    async def read(self, *names: str) -> dict[str, MotorState]:
        """
        Read the current state of one or more motors.

        Parameters
        ----------
        *names : str
            One or more motor names to query.

        Returns
        -------
        dict[str, MotorState]
            Mapping of motor name to its current MotorState snapshot.

        Raises
        ------
        KeyError
            If any name is not in the valid motor list.

        Examples
        --------
        >>> states = await accessor.read("Sample X", "Sample Y")
        >>> print(states["Sample X"].position)
        """
        self._validate_motors(names)
        response = await self._conn.get_motor(motors=list(names))
        return {
            entry["motor"]: MotorState(
                position=entry["position"],
                goal=entry["goal"],
                status=entry["status"],
                time=entry["time"],
            )
            for entry in response["data"]
        }

    async def set(
        self,
        name: str,
        value: float,
        *,
        command: Command = "Backlash Move",
    ) -> None:
        """
        Command a single motor to a target position.

        Parameters
        ----------
        name : str
            Motor name to move.
        value : float
            Target position.
        command : Command, optional
            BCS command string, by default "Backlash Move".

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If `name` is not a valid motor.
        ValueError
            If `command` is not a valid BCS command string.

        Examples
        --------
        >>> await accessor.set("Sample X", 5.0)
        >>> await accessor.set("Sample X", 5.0, command="Normal Move")
        """
        self._validate_motors((name,))
        self._validate_command(command)
        await self._conn.command_motor(
            commands=[command],
            motors=[name],
            goals=[value],
        )

    async def set_many(
        self,
        targets: dict[str, float],
        *,
        command: Command = "Backlash Move",
    ) -> None:
        """
        Command multiple motors to target positions simultaneously.

        Parameters
        ----------
        targets : dict[str, float]
            Mapping of motor name to target position.
        command : Command, optional
            BCS command string applied to all motors, by default "Backlash Move".

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If any motor name in `targets` is not valid.
        ValueError
            If `command` is not a valid BCS command string.

        Examples
        --------
        >>> await accessor.set_many({"Sample X": 5.0, "Sample Y": -2.5})
        """
        self._validate_motors(list(targets.keys()))
        self._validate_command(command)
        names = list(targets.keys())
        await self._conn.command_motor(
            commands=[command] * len(names),
            motors=names,
            goals=list(targets.values()),
        )

    async def wait(
        self,
        names: list[str],
        timeout: float = 30.0,
        abort: AbortFlag | None = None,
    ) -> None:
        """
        Wait for one or more motors to reach their target positions.

        Parameters
        ----------
        names : list[str]
            Motor names to wait on.
        timeout : float, optional
            Maximum seconds to wait before raising MotorTimeoutError, by default 30.0.
        abort : AbortFlag or None, optional
            If provided, the wait will raise ScanAbortedError when the flag is set.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If any name is not a valid motor.
        MotorTimeoutError
            If any motor does not reach its target within `timeout` seconds.
        ScanAbortedError
            If `abort` flag is set during the wait.

        Notes
        -----
        Delegates to `wait_for_motors` from `resonance.api.core.primitives`.
        Pass an `AbortFlag` to enable cooperative cancellation from Jupyter
        or programmatic interfaces: set the flag from another task/cell to
        stop waiting and raise `ScanAbortedError`.

        Examples
        --------
        >>> flag = AbortFlag()
        >>> await accessor.set_many({"Sample X": 10.0, "Sample Y": 5.0})
        >>> await accessor.wait(["Sample X", "Sample Y"], timeout=60.0, abort=flag)
        """
        # TODO: expose check_interval as a parameter for fine-grained polling control
        self._validate_motors(names)
        await wait_for_motors(
            server=self._conn,
            motors=names,
            timeout=timeout,
            abort_flag=abort,
        )
