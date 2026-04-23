from __future__ import annotations

from enum import IntFlag
from typing import Any


class MotorStatus(IntFlag):
    HOME = 1
    FORWARD_LIMIT = 2
    REVERSE_LIMIT = 4
    MOTOR_DIRECTION = 8
    MOTOR_OFF = 16
    MOVE_COMPLETE = 32
    FOLLOWING_ERROR = 64
    NOT_IN_DEAD_BAND = 128

    def is_set(self, flag: MotorStatus) -> bool:
        return bool(self & flag)


class BCSServer:
    async def connect(self, **_: Any) -> None:
        return None

    async def bcs_request(
        self, command_name: str, param_dict: dict[str, Any], debugging: bool = False
    ) -> dict[str, Any]:
        raise RuntimeError(
            f"BCS backend unavailable for command {command_name!r}; install runtime BCS client"
        )

    async def acquire_data(self, chans: list[str], time: float) -> dict[str, Any]:
        return await self.bcs_request("AcquireData", {"chans": chans, "time": time})

    async def get_acquired_array(self, chans: list[str]) -> dict[str, Any]:
        return await self.bcs_request("GetAcquiredArray", {"chans": chans})

    async def get_instrument_driver_status(self, name: str) -> dict[str, Any]:
        return await self.bcs_request("GetInstrumentDriverStatus", {"name": name})

    async def start_instrument_driver(self, name: str) -> dict[str, Any]:
        return await self.bcs_request("StartInstrumentDriver", {"name": name})

    async def start_instrument_acquire(
        self, name: str, run_type: str, acq_time_s: float
    ) -> dict[str, Any]:
        return await self.bcs_request(
            "StartInstrumentAcquire",
            {"name": name, "run_type": run_type, "acq_time_s": acq_time_s},
        )

    async def get_di(self, chans: list[str]) -> dict[str, Any]:
        return await self.bcs_request("GetDI", {"chans": chans})

    async def set_do(self, chan: str, value: bool) -> dict[str, Any]:
        return await self.bcs_request("SetDO", {"chan": chan, "value": value})

    async def get_motor(self, motors: list[str]) -> dict[str, Any]:
        return await self.bcs_request("GetMotor", {"motors": motors})

    async def command_motor(
        self, commands: list[str], motors: list[str], goals: list[float]
    ) -> dict[str, Any]:
        return await self.bcs_request(
            "CommandMotor",
            {"commands": commands, "motors": motors, "goals": goals},
        )


__all__ = ["BCSServer", "MotorStatus"]
