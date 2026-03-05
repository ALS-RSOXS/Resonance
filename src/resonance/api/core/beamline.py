from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

from bcs import BCSz
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from resonance.api.core.ai import AIAccessor
from resonance.api.core.det import AreaDetector
from resonance.api.core.dio import DIOAccessor
from resonance.api.core.motors import MotorAccessor
from resonance.api.core.scan import ScanExecutor, ScanPlan

if TYPE_CHECKING:
    import pandas as pd

    from resonance.api.data.writer import RunWriter


class Connection(BaseSettings):
    """
    Connection settings loaded from environment variables.

    Reads BCS_SERVER_ADDRESS and BCS_SERVER_PORT from the environment
    or a .env file. Used by `Beamline.create()`.

    Parameters
    ----------
    addr : str
        BCS server hostname or IP address (env: BCS_SERVER_ADDRESS).
        Default: "localhost".
    port : int
        BCS server port (env: BCS_SERVER_PORT).
        Default: 5577.
    """

    addr: str = Field(default="localhost", alias="BCS_SERVER_ADDRESS")
    port: int = Field(default=5577, alias="BCS_SERVER_PORT")
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Beamline:
    """
    High-level interface for beamline hardware control.

    Composes a BCSz server connection with typed accessors for motors,
    analog inputs, and digital I/O, plus a scan executor for running
    DataFrame-defined scans. Does not subclass BCSz.

    Parameters
    ----------
    conn : BCSz.BCSServer
        Connected BCS server instance. Use `Beamline.create()` to
        construct with automatic connection from environment variables.

    Attributes
    ----------
    ai : AIAccessor
        Read and acquire analog input channels.
    motors : MotorAccessor
        Read, move, and wait for motors.
    dio : DIOAccessor
        Read and set digital I/O channels (e.g. shutter).

    Examples
    --------
    >>> bl = await Beamline.create()
    >>> data = await bl.ai.trigger_and_read(["Photodiode"], acquisition_time=1.0)
    >>> await bl.motors.set("Sample X", 10.5)
    >>> results = await bl.scan_from_dataframe(scan_df, ai_channels=["Photodiode"])
    """

    def __init__(self, conn: BCSz.BCSServer) -> None:
        self._conn = conn
        self.ai = AIAccessor(conn)
        self.motors = MotorAccessor(conn)
        self.dio = DIOAccessor(conn)
        self.detector = AreaDetector(conn)
        self._executor = ScanExecutor(conn)
        # TODO: add optional detector setup (e.g. CCD warm-up, status check) here
        # TODO: add future EPICS/Bluesky adapter hook when migrating from BCSz

    @classmethod
    async def create(cls) -> Beamline:
        """
        Create and connect a Beamline from environment variables.

        Reads BCS_SERVER_ADDRESS and BCS_SERVER_PORT from the environment or
        a .env file, creates a BCSz server, and connects.

        Returns
        -------
        Beamline
            A connected, ready-to-use Beamline instance.

        Raises
        ------
        ConnectionError
            If the BCS server is unreachable or connection fails.
        """
        config = Connection()
        server = BCSz.BCSServer()
        buff = io.StringIO()
        with redirect_stdout(buff):
            await server.connect(**config.model_dump())
        return cls(server)

    async def scan_from_dataframe(
        self,
        df: pd.DataFrame,
        ai_channels: list[str] | None = None,
        default_delay: float = 0.1,
        shutter: str = "Shutter Output",
        motor_timeout: float = 30.0,
        progress: bool = True,
        actuate_every: bool = False,
        writer: RunWriter | None = None,
        with_detector: bool = False,
    ) -> pd.DataFrame:
        """
        Execute a scan defined by a DataFrame.

        Each row defines one scan point: motor columns set motor positions,
        and an optional exposure column sets per-point acquisition time.

        Parameters
        ----------
        df : pd.DataFrame
            Scan definition. Motor columns must match valid motor names.
            An optional "exposure" (or "exp", "count_time") column sets
            per-point acquisition time.
        ai_channels : list[str] or None, optional
            AI channels to acquire at each point. If None, uses
            ["Photodiode", "TEY signal", "AI 3 Izero"].
        default_delay : float, optional
            Settle delay after each motor move in seconds (default: 0.1).
        shutter : str, optional
            DIO channel name for the shutter (default: "Shutter Output").
        motor_timeout : float, optional
            Maximum wait time for motor moves in seconds (default: 30.0).
        progress : bool, optional
            Show a tqdm progress bar (default: True).
        actuate_every : bool, optional
            If True, open/close the shutter per point. If False (default),
            open the shutter once for the entire scan.
        writer : RunWriter or None, optional
            If provided, scan data are persisted to the beamtime SQLite database
            via the writer. The caller is responsible for constructing and
            opening the writer before passing it here.
        with_detector : bool, optional
            If True, acquire a 2D detector image at each scan point using the
            beamline's AreaDetector. Requires a writer for image persistence.
            Shutter actuation is hardware-driven (default: False).

        Returns
        -------
        pd.DataFrame
            Results with columns: motor_position, channel_mean, channel_std,
            exposure, timestamp per row.

        Notes
        -----
        To abort a running scan from another Jupyter cell, call
        ``await bl.abort_scan()`` while the scan task is running.
        """
        scan_plan = ScanPlan.from_dataframe(
            df,
            ai_channels=ai_channels,
            default_delay=default_delay,
            shutter=shutter,
            actuate_every=actuate_every,
        )
        return await self._executor.execute_scan(
            scan_plan,
            progress=progress,
            writer=writer,
            detector=self.detector if with_detector else None,
        )

    async def abort_scan(self) -> None:
        """
        Request an abort of the currently running scan.

        Sets the internal abort flag. The scan stops after the current
        point completes and returns partial results.

        Notes
        -----
        Safe to call even when no scan is running (no-op).
        From Jupyter: run the scan as an asyncio.Task and call this
        method from another cell while it executes.
        Programmatic: call from any async context.
        """
        await self._executor.abort()

    @property
    def is_scanning(self) -> bool:
        """
        Whether a scan is currently running.

        Returns
        -------
        bool
            True if a scan is in progress, False otherwise.
        """
        return self._executor.current_scan is not None
