from typing import Any

def decode_z85(data: str) -> bytes: ...
def decode_z85_parallel(data: str) -> bytes: ...

class BcsConnection:
    def __init__(
        self,
        addr: str,
        port: int,
        recv_timeout_ms: int | None = ...,
        send_timeout_ms: int | None = ...,
        use_curve: bool = ...,
    ) -> None: ...
    def bcs_request(
        self, command_name: str, params: dict[str, Any]
    ) -> dict[str, Any]: ...
