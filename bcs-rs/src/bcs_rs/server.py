import asyncio
from typing import Any

from bcs_rs._core import BcsConnection


class BCSServer:
    def __init__(self, addr: str = "127.0.0.1", port: int = 5577) -> None:
        self._addr = addr
        self._port = port
        self._conn: BcsConnection | None = None

    async def connect(
        self,
        addr: str | None = None,
        port: int | None = None,
        recv_timeout_ms: int = 5000,
        send_timeout_ms: int = 5000,
    ) -> None:
        if addr is not None:
            self._addr = addr
        if port is not None:
            self._port = port
        self._conn = await asyncio.to_thread(
            BcsConnection,
            self._addr,
            self._port,
            recv_timeout_ms,
            send_timeout_ms,
        )

    async def bcs_request(
        self,
        command_name: str,
        param_dict: dict[str, Any],
        debugging: bool = False,
    ) -> dict[str, Any]:
        if self._conn is None:
            raise RuntimeError("BCSServer not connected")
        if debugging:
            print(f"API command {command_name} BEGIN.")
        result = await asyncio.to_thread(
            self._conn.bcs_request,
            command_name,
            param_dict,
        )
        out = dict(result)
        if debugging and "API_delta_t" in out:
            print(f"API command {command_name} END {out['API_delta_t']} s.")
        return out
