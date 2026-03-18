import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from bcs_rs._core import BcsConnection

if sys.platform[:3] == "win":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _connect(
    addr: str,
    port: int,
    recv_timeout_ms: int,
    send_timeout_ms: int,
    use_curve: bool,
) -> BcsConnection:
    return BcsConnection(addr, port, recv_timeout_ms, send_timeout_ms, use_curve)


def _bcs_request(
    conn: BcsConnection,
    command_name: str,
    param_dict: dict[str, Any],
) -> dict[str, Any]:
    return conn.bcs_request(command_name, param_dict)


class BCSServer:
    def __init__(self, addr: str = "127.0.0.1", port: int = 5577) -> None:
        self._addr = addr
        self._port = port
        self._conn: BcsConnection | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def connect(
        self,
        addr: str | None = None,
        port: int | None = None,
        recv_timeout_ms: int = 5000,
        send_timeout_ms: int = 5000,
        use_curve: bool = True,
    ) -> None:
        if addr is not None:
            self._addr = addr
        if port is not None:
            self._port = port
        loop = asyncio.get_event_loop()
        self._conn = await loop.run_in_executor(
            self._executor,
            _connect,
            self._addr,
            self._port,
            recv_timeout_ms,
            send_timeout_ms,
            use_curve,
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
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            _bcs_request,
            self._conn,
            command_name,
            param_dict,
        )
        out = dict(result)
        if debugging and "API_delta_t" in out:
            print(f"API command {command_name} END {out['API_delta_t']} s.")
        return out
