"""Scan orchestration and execution using core primitives."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

try:
    from tqdm.asyncio import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

import numpy as np
import pandas as pd
from uncertainties import ufloat

from resonance.api.core.primitives import (
    AbortFlag,
    motor_move,
    shutter_control,
    wait_for_settle,
)
from resonance.api.header_map import normalize_header
from resonance.api.types import ScanAbortedError, ScanPoint, ScanResult
from resonance.api.validation import validate_scan_dataframe

if TYPE_CHECKING:
    from collections.abc import Iterator

    from bcs import BCSz
    from uncertainties import Variable

    from resonance.api.core.det import AreaDetector
    from resonance.api.data.writer import RunWriter


class ScanPlan:
    """
    Validated scan plan constructed from a list of points or a DataFrame.

    Parameters
    ----------
    points : list[ScanPoint]
        Ordered sequence of scan points.
    motor_names : list[str]
        Names of all motors referenced in the scan.
    ai_channels : list[str]
        Analog input channel names to acquire at each point.
    shutter : str
        DIO channel name of the light shutter.
    actuate_every : bool
        If True, the shutter is opened and closed per point.
        If False, the shutter is opened once for the entire scan.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "Sample X": [0, 10, 20],
    ...     "Sample Y": [0, 0, 0],
    ...     "exposure": [0.1, 0.1, 0.1],
    ... })
    >>> scan_plan = ScanPlan.from_dataframe(df, ai_channels=["Photodiode"])
    """

    def __init__(
        self,
        points: list[ScanPoint],
        motor_names: list[str],
        ai_channels: list[str],
        shutter: str = "Light Output",
        actuate_every: bool = False,
    ) -> None:
        self.points = points
        self.motor_names = motor_names
        self.ai_channels = ai_channels
        self.shutter = shutter
        self.actuate_every = actuate_every

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        ai_channels: list[str] | None = None,
        default_exposure: float = 1.0,
        default_delay: float = 0.2,
        shutter: str = "Light Output",
        actuate_every: bool = False,
    ) -> ScanPlan:
        """
        Build a validated scan plan from a DataFrame.

        Each row becomes one `ScanPoint`. Motor columns are detected via
        `validate_scan_dataframe`; an optional exposure column is also detected.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame whose columns are motor names and optionally an exposure
            time column.
        ai_channels : list[str] or None
            Channels to acquire at each point.  If None, a beamline default
            set is used.
        default_exposure : float
            Exposure time in seconds applied when the DataFrame has no exposure
            column.
        default_delay : float
            Settle delay in seconds after each motor move.
        shutter : str or None
            DIO channel name of the light shutter. If None, the shutter is not used
        actuate_every : bool
            Per-point shutter mode.

        Returns
        -------
        ScanPlan
            Fully validated scan plan.

        Raises
        ------
        ValidationError
            If the DataFrame fails structural or value validation.
        """
        motor_cols, exposure_col = validate_scan_dataframe(df)

        # TODO: make default AI channels configurable at the Beamline level
        if ai_channels is None:
            ai_channels = ["Photodiode", "TEY signal", "AI 3 Izero"]

        points: list[ScanPoint] = []
        for idx, row in df.iterrows():
            motors = {col: float(row[col]) for col in motor_cols}
            exposure = float(row[exposure_col]) if exposure_col else default_exposure
            point = ScanPoint(
                index=int(idx),  # type: ignore[arg-type]
                motors=motors,
                exposure_time=exposure,
                ai_channels=ai_channels,
                delay_after_move=default_delay,
            )
            point.validate()
            points.append(point)

        return cls(
            points=points,
            motor_names=motor_cols,
            ai_channels=ai_channels,
            shutter=shutter,
            actuate_every=actuate_every,
        )

    def estimated_duration_seconds(
        self,
        motor_time: float = 0.1,
        api_time: float = 0.5,
    ) -> float:
        """
        Compute estimated total scan duration.

        Per-point cost: motor_time + api_time + exposure_time + delay_after_move.

        Parameters
        ----------
        motor_time : float
            Expected motor-move overhead per point in seconds.
        api_time : float
            Expected API round-trip overhead per point in seconds.

        Returns
        -------
        float
            Total estimated duration in seconds.

        Notes
        -----
        Does not account for shutter actuation overhead when ``actuate_every=True``.
        """
        # TODO: account for shutter actuation time when actuate_every=True
        return sum(
            motor_time + api_time + p.exposure_time + p.delay_after_move
            for p in self.points
        )

    def describe(
        self,
        motor_time: float = 0.1,
        api_time: float = 0.5,
    ) -> None:
        """
        Print a human-readable summary of the scan plan.

        Displays point count, unique motor values, and estimated duration.

        Parameters
        ----------
        motor_time : float
            Motor-move overhead per point in seconds used for the estimate.
        api_time : float
            API overhead per point in seconds used for the estimate.
        """
        lines: list[str] = [
            f"Scan plan: {len(self.points)} points, {len(self.motor_names)} motors",
            "",
            "Unique values:",
        ]
        for m in self.motor_names:
            n = len({p.motors[m] for p in self.points})
            lines.append(f"  {m}: {n}")
        lines.append(f"  exposure: {len({p.exposure_time for p in self.points})}")
        delay_val = self.points[0].delay_after_move if self.points else 0.0
        lines.append(
            f"  delay: {len({p.delay_after_move for p in self.points})} ({delay_val} s)"
        )
        lines.append("")
        duration = self.estimated_duration_seconds(
            motor_time=motor_time, api_time=api_time
        )
        minutes, hours = duration / 60.0, duration / 3600.0
        if hours >= 1:
            time_str = f"{hours:.1f} h ({duration:.0f} s)"
        elif minutes >= 1:
            time_str = f"{minutes:.1f} min ({duration:.0f} s)"
        else:
            time_str = f"{duration:.1f} s"
        lines.append(f"Estimated duration: {time_str}")
        lines.append(
            f"  per point: {motor_time} s (motor) + {api_time} s (api)"
            f" + exposure + {delay_val} s (delay)"
        )
        print("\n".join(lines))

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[ScanPoint]:
        return iter(self.points)


class ScanExecutor:
    """
    Executes `ScanPlan` instances against a live BCS server.

    Supports two interrupt modes:

    1. Programmatic abort: call ``await executor.abort()`` from any async
       context to set the abort flag.  The next check inside `execute_point`
       or `wait_for_settle` raises `ScanAbortedError`.

    2. Jupyter / IPython interrupt: create the scan as an ``asyncio.Task``
       and call ``await bl.abort_scan()`` from another cell to set the abort
       flag and stop after the current point.

    ``asyncio.CancelledError`` is raised when the Task is cancelled via
    ``task.cancel()``.  ``execute_scan`` catches this, sets the abort flag,
    and returns any partial results already collected.  If no results have
    been collected the error is re-raised.

    Parameters
    ----------
    conn : BCSz.BCSServer
        Active BCS server connection.
    """

    def __init__(self, conn: BCSz.BCSServer) -> None:
        self._conn = conn
        self._abort_flag = AbortFlag()
        self._current_scan: ScanPlan | None = None

    @property
    def current_scan(self) -> ScanPlan | None:
        """
        Currently running scan plan.

        Returns
        -------
        ScanPlan or None
            The active `ScanPlan` during execution, or ``None`` when idle.
        """
        return self._current_scan

    async def abort(self) -> None:
        """
        Request abort of the running scan.

        Sets the internal `AbortFlag`.  The scan loop will raise
        `ScanAbortedError` at the next abort-check site (start of
        `execute_point`, inside `wait_for_settle`, or inside
        `wait_for_motors`).

        Notes
        -----
        Safe to call from a separate task or thread-pool executor while the
        scan is running.
        """
        await self._abort_flag.set()

    async def execute_point(
        self,
        point: ScanPoint,
        motor_timeout: float = 30.0,
        restore_motors: bool = False,
        use_shutter: bool = True,
        detector: AreaDetector | None = None,
    ) -> ScanResult:
        """
        Execute a single scan point.

        Parameters
        ----------
        point : ScanPoint
            The point to execute.
        motor_timeout : float
            Maximum time in seconds to wait for motors to reach position.
        restore_motors : bool
            If True, motor positions are restored to their pre-move values
            after the point completes.
        use_shutter : bool
            If True, the shutter is opened and closed around acquisition.
            Pass False when the caller already holds the shutter open.
        detector : AreaDetector or None, optional
            If provided, a 2D detector image is acquired after AI acquisition using
            the point's exposure_time. Shutter actuation is hardware-driven; no
            plan-level shutter wraps this call.

        Returns
        -------
        ScanResult
            Measured values with per-channel `ufloat` statistics.

        Raises
        ------
        ScanAbortedError
            If the abort flag is set before or during execution.
        MotorError
            If a motor move or restore fails.
        """
        if await self._abort_flag.is_set():
            raise ScanAbortedError("Scan aborted before point execution")

        async with motor_move(
            self._conn,
            point.motors,
            timeout=motor_timeout,
            restore_on_exit=restore_motors,
        ):
            await wait_for_settle(point.delay_after_move, self._abort_flag)

            channels = point.ai_channels or (
                self._current_scan.ai_channels if self._current_scan else []
            )

            async def _acquire() -> dict[str, Any]:
                if await self._abort_flag.is_set():
                    raise ScanAbortedError("Scan aborted before acquisition")
                await self._conn.acquire_data(chans=channels, time=point.exposure_time)  # pyright: ignore[reportArgumentType]
                return await self._conn.get_acquired_array(chans=channels)

            if use_shutter:
                shutter_name = (
                    self._current_scan.shutter if self._current_scan else "Light Output"
                )
                async with shutter_control(self._conn, shutter=shutter_name):
                    result = await _acquire()
            else:
                result = await _acquire()

        ai_data: dict[str, Variable] = {}
        raw_data: dict[str, list[float]] = {}
        for chan_data in result["chans"]:
            name: str = chan_data["chan"]
            data = np.array(chan_data["data"], dtype=float)
            raw_data[name] = data.tolist()
            if data.size == 0:
                ai_data[name] = ufloat(np.nan, np.nan)
            else:
                mean = float(np.nanmean(data))
                std_err = float(np.nanstd(data, ddof=1) / np.sqrt(data.size))
                ai_data[name] = ufloat(mean, std_err)

        scan_result = ScanResult(
            index=point.index,
            motors=point.motors,
            ai_data=ai_data,
            exposure_time=point.exposure_time,
            timestamp=time.time(),
            raw_data=raw_data,
        )
        if detector is not None:
            scan_result.image = await detector.acquire(point.exposure_time)
        return scan_result

    async def execute_scan(
        self,
        scan_plan: ScanPlan,
        progress: bool = True,
        writer: RunWriter | None = None,
        detector: AreaDetector | None = None,
    ) -> pd.DataFrame:
        """
        Execute a complete scan plan and return results as a DataFrame.

        The shutter behaviour depends on `scan_plan.actuate_every`:

        - ``False`` (default): the shutter is opened once before the first
          point and closed after the last point (or on abort).
        - ``True``: the shutter is actuated individually for every point via
          `execute_point`.

        Interrupt modes
        ---------------
        Programmatic: call ``await executor.abort()`` from any async context.
        The `AbortFlag` is checked at the start of each point;
        `ScanAbortedError` is raised and partial results are returned.

        Jupyter / IPython: create this coroutine as an ``asyncio.Task`` and
        call ``await bl.abort_scan()`` from another cell to set the abort
        flag and stop after the current point.

        ``asyncio.CancelledError``: raised when the Task is cancelled via
        ``task.cancel()``.  Partial results are returned if any points
        completed; otherwise the error is re-raised.

        Parameters
        ----------
        scan_plan : ScanPlan
            Validated scan plan to execute.
        progress : bool
            If True and tqdm is installed, show an async progress bar.
            Falls back to simple per-point print statements.
        writer : RunWriter or None, optional
            If provided, scalar scan data (motor positions, AI means, exposure,
            timestamps) are written to the open beamtime database. The writer
            must not have an open run before this method is called; it will
            call open_run, open_stream, write_event, and close_run internally.
        detector : AreaDetector or None, optional
            If provided, a 2D image is acquired at each scan point and written to
            the "detector_image" field in the primary stream. Requires writer to
            be set for persistence. Shutter is hardware-driven.

        Returns
        -------
        pd.DataFrame
            One row per completed point with motor positions, per-channel
            mean/std columns, exposure time, and timestamp.

        Notes
        -----
        # TODO: add Bluesky-compatible start/stop document emission once RunWriter is stable
        """
        self._current_scan = scan_plan
        await self._abort_flag.clear()

        if writer is not None:
            data_keys: dict[str, dict[str, str]] = {
                **{
                    f"{normalize_header(m)}_position": {
                        "dtype": "number",
                        "units": "mm",
                        "source": "motor",
                    }
                    for m in scan_plan.motor_names
                },
                **{
                    normalize_header(ch): {
                        "dtype": "number",
                        "units": "V",
                        "source": "ai",
                    }
                    for ch in scan_plan.ai_channels
                },
                normalize_header("exposure"): {
                    "dtype": "number",
                    "units": "s",
                    "source": "plan",
                },
            }
            if detector is not None:
                data_keys["detector_image"] = detector.describe()
            # TODO: propagate plan_name from ScanPlan once the attribute is added
            writer.open_run("scan")
            writer.open_stream("primary", data_keys)

        results: list[ScanResult] = []

        if progress and _TQDM_AVAILABLE:
            iterator = _tqdm(scan_plan.points, desc="Scanning", unit="pt")
        else:
            if progress and not _TQDM_AVAILABLE:
                print("tqdm not available, showing simple progress")
            progress = False
            iterator = iter(scan_plan.points)

        async def _run_points() -> None:
            for i, point in enumerate(iterator):
                if not progress:
                    print(f"Point {i + 1}/{len(scan_plan.points)}", end="\r")
                result = await self.execute_point(
                    point,
                    restore_motors=False,
                    use_shutter=scan_plan.actuate_every,
                    detector=detector,
                )
                results.append(result)
                if writer is not None:
                    event_data: dict[str, float | int | str | bool] = {
                        **{
                            f"{normalize_header(m)}_position": v
                            for m, v in result.motors.items()
                        },
                        **{
                            normalize_header(ch): result.ai_data[ch].nominal_value
                            for ch in result.ai_data
                        },
                        normalize_header("exposure"): result.exposure_time,
                    }
                    timestamps: dict[str, float] = {
                        **{
                            f"{normalize_header(m)}_position": result.timestamp
                            for m in result.motors
                        },
                        **{
                            normalize_header(ch): result.timestamp
                            for ch in result.ai_data
                        },
                        normalize_header("exposure"): result.timestamp,
                    }
                    event_uid = writer.write_event(event_data, timestamps)
                    if result.image is not None:
                        writer.write_image(event_uid, "detector_image", result.image)

        _exit_status = "success"
        try:
            if scan_plan.actuate_every:
                await _run_points()
            else:
                async with shutter_control(self._conn, shutter=scan_plan.shutter):
                    await _run_points()

        except ScanAbortedError:
            _exit_status = "aborted"
            print(f"\nScan aborted after {len(results)}/{len(scan_plan.points)} points")
            if not results:
                raise

        except asyncio.CancelledError:
            _exit_status = "aborted"
            await self._abort_flag.set()
            print(
                f"\nScan interrupted after {len(results)}/{len(scan_plan.points)} points"
            )
            if not results:
                raise
            # Intentionally not re-raising: returns partial results to the caller.
            # Task cancellation propagation is sacrificed for interactive usability.

        finally:
            self._current_scan = None
            if writer is not None:
                writer.close_run(exit_status=_exit_status)
            if not progress:
                print()

        return pd.DataFrame([r.to_series() for r in results])
