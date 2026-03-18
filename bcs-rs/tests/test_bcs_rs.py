import pytest

from bcs_rs import BCSServer, decode_z85, decode_z85_parallel


def test_decode_z85_roundtrip():
    z85_alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
    chunk = z85_alphabet[:20]
    assert len(chunk) == 20
    raw = decode_z85(chunk)
    assert len(raw) == 16


def test_decode_z85_parallel_same_result():
    z85_alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
    chunk = z85_alphabet[:20]
    a = decode_z85(chunk)
    b = decode_z85_parallel(chunk)
    assert a == b


def test_bcsserver_bcs_request_raises_when_not_connected():
    server = BCSServer("127.0.0.1", 5577)
    with pytest.raises(RuntimeError, match="not connected"):
        import asyncio
        asyncio.run(server.bcs_request("SomeCommand", {}))
