from __future__ import annotations

import numpy as np
import pytest

from resonance.api.data.compression import compress_array, decompress_array


def test_compress_decompress_zlib() -> None:
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    buf, method = compress_array(arr, method="zlib")
    assert method == "zlib"
    result = decompress_array(buf, method="zlib", dtype="uint16", shape=(2, 2))
    assert np.array_equal(result, arr)


def test_compress_none() -> None:
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    buf, method = compress_array(arr, method="none")
    assert method == "none"
    assert len(buf) == arr.nbytes


def test_decompress_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        decompress_array(b"x", method="lz4", dtype="uint16", shape=(1, 1))


def test_compress_large_array() -> None:
    arr = np.zeros((256, 256), dtype=np.float32)
    buf, method = compress_array(arr, method="zlib")
    result = decompress_array(buf, method=method, dtype="float32", shape=(256, 256))
    assert np.array_equal(result, arr)


def test_decompress_nd_shape() -> None:
    arr = np.ones((4, 8, 16), dtype=np.uint8)
    buf, method = compress_array(arr, method="zlib")
    result = decompress_array(buf, method=method, dtype="uint8", shape=(4, 8, 16))
    assert result.shape == (4, 8, 16)
    assert np.array_equal(result, arr)
