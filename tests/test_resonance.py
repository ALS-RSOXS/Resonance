"""Smoke tests for resonance package."""


def test_import_resonance() -> None:
    """Resonance package can be imported."""
    import resonance

    assert resonance.__version__ == "0.1.0"


def test_import_resonance_api() -> None:
    """Resonance API subpackage can be imported."""
    import resonance.api

    assert hasattr(resonance.api, "RsoxsServer")
    assert hasattr(resonance.api, "Beamline")
    assert hasattr(resonance.api, "MotorAccessor")
    assert hasattr(resonance.api, "AIAccessor")
    assert hasattr(resonance.api, "DIOAccessor")


def test_import_resonance_mcp() -> None:
    """Resonance MCP subpackage can be imported."""
    import resonance.mcp

    assert resonance.mcp.__name__ == "resonance.mcp"
