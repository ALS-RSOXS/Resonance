from __future__ import annotations

import zlib
from typing import Literal

import numpy as np


def compress_array(
    data: np.ndarray,
    *,
    method: Literal["zlib", "none"] = "zlib",
) -> tuple[bytes, str]:
    """Compress a NumPy array to bytes.

    Parameters
    ----------
    data : np.ndarray
        Array to compress. Any dtype and shape.
    method : {"zlib", "none"}, optional
        Compression algorithm. The returned method string is round-trippable
        with ``decompress_array``.

    Returns
    -------
    tuple[bytes, str]
        Compressed (or raw) bytes and the method string used.
    """
    if method == "zlib":
        return zlib.compress(data.tobytes(), 6), "zlib"
    return data.tobytes(), "none"


def decompress_array(
    buffer: bytes,
    *,
    method: str,
    dtype: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Decompress bytes back to a NumPy array.

    Parameters
    ----------
    buffer : bytes
        Compressed or raw bytes produced by ``compress_array``.
    method : str
        Compression method used during compression. Must be ``"zlib"`` or
        ``"none"``.
    dtype : str
        NumPy dtype string (e.g. ``"uint16"``, ``"float32"``) describing the
        array element type.
    shape : tuple[int, ...]
        Target shape of the output array. Accepts arbitrary N-D shapes.

    Returns
    -------
    np.ndarray
        Reconstructed array with the given dtype and shape.

    Raises
    ------
    ValueError
        If ``method`` is not a recognised compression method.
    """
    if method == "zlib":
        raw = zlib.decompress(buffer)
    elif method == "none":
        raw = buffer
    else:
        raise ValueError(f"Unknown compression method: {method!r}")
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)
