"""Canonical header names for motors, AI channels, and related fields."""

from __future__ import annotations

HEADER_NAMES: dict[str, str] = {
    "fits_index": "fits_index",
    "Beamline Energy": "energy",
    "EPU Polarization": "polarization",
    "Sample Theta": "sam_th",
    "CCD Theta": "det_th",
    "Sample X": "sam_x",
    "Sample Y": "sam_y",
    "Sample Z": "sam_z",
    "image": "raw_image",
    "EXPOSURE": "exposure",
    "Higher Order Suppressor": "hos",
    "Upstream JJ Vert Aperture": "slits_vert",
    "Upstream JJ Horz Aperture": "slits_horz",
    "Beam Current": "beam_current",
    "AI 3 Izero": "i0",
}


def normalize_header(name: str) -> str:
    """
    Map a header name to its canonical form.

    Uses HEADER_NAMES when present; otherwise replaces spaces with
    underscores and lowercases the string.

    Parameters
    ----------
    name : str
        Original header (e.g. motor name, AI channel, or column name).

    Returns
    -------
    str
        Canonical name (e.g. sam_x, i0, exposure).
    """
    if name in HEADER_NAMES:
        return HEADER_NAMES[name]
    return name.replace(" ", "_").lower()
