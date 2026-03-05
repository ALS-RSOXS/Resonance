"""Connection management for Beamline in MCP context."""

import asyncio
from typing import Optional

from resonance.api.core import Beamline


class ConnectionManager:
    """Singleton connection manager for Beamline."""

    _instance: Optional["ConnectionManager"] = None
    _lock = asyncio.Lock()
    _server: Beamline | None = None
    _connected: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_server(self) -> Beamline:
        """
        Get or create Beamline instance.

        Returns
        -------
        Beamline
            Connected beamline instance

        Raises
        ------
        ConnectionError
            If connection fails
        """
        async with self._lock:
            if self._server is None or not self._connected:
                self._server = await Beamline.create()
                self._connected = True
            return self._server

    async def ensure_connected(self) -> None:
        """
        Ensure beamline is connected, reconnect if needed.

        Raises
        ------
        ConnectionError
            If reconnection fails
        RuntimeError
            If connection state is inconsistent
        """
        async with self._lock:
            if self._server is None or not self._connected:
                try:
                    self._server = await Beamline.create()
                    self._connected = True
                except ConnectionError as e:
                    self._connected = False
                    self._server = None
                    raise ConnectionError(
                        f"Failed to connect to beamline server: {e}"
                    ) from e
                except Exception as e:
                    self._connected = False
                    self._server = None
                    raise RuntimeError(
                        f"Unexpected error connecting to beamline server: {e}"
                    ) from e

    async def disconnect(self) -> None:
        """Disconnect from beamline."""
        async with self._lock:
            self._connected = False
            self._server = None

    @property
    def is_connected(self) -> bool:
        """Check if beamline is connected."""
        return self._connected and self._server is not None


connection_manager = ConnectionManager()
