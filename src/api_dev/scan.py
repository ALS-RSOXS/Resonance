"""Scan orchestration and execution"""

import time
from typing import List, Optional

import numpy as np
import pandas as pd
from bcs.BCSz import BCSServer
from uncertainties import ufloat

from .core import AbortFlag, motor_move, shutter_control, wait_for_settle
from .types import Instrument, ScanAbortedError, ScanPoint, ScanResult
from .validation import validate_scan_dataframe

MOTOR_MOVE_TIME_PER_POINT = 0.1
API_PROCESSING_TIME_PER_POINT = 0.5


# ============================================================================
# Scan Plan
# ============================================================================


class ScanPlan:
    """Validated scan plan built from DataFrame"""

    def __init__(
        self,
        points: List[ScanPoint],
        motor_names: List[str],
        ai_channels: List[str],
        shutter: str = "Light Output",
        instrument: Instrument = "Photodiode",
        actuate_every: bool = True,
    ):
        self.points = points
        self.motor_names = motor_names
        self.ai_channels = ai_channels
        self.shutter = shutter
        self.actuate_every = actuate_every

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        ai_channels: Optional[List[str]] = None,
        default_exposure: float = 1.0,
        default_delay: float = 0.2,
        shutter: str = "Light Output",
        instrument: Instrument = "Photodiode",
        actuate_every: bool = False,
    ) -> "ScanPlan":
        """
        Build validated scan plan from DataFrame.

        Parameters:
            df: DataFrame with motor columns and optional exposure column
            ai_channels: AI channels to read (None = will use defaults)
            default_exposure: Exposure time if no column found
            default_delay: Delay after motor move
            shutter: Shutter channel name
            actuate_every: If True, open/close shutter per point; if False, open once for whole scan

        Returns:
            Validated ScanPlan

        Raises:
            ValidationError: If DataFrame validation fails
        """
        # Validate DataFrame
        motor_cols, exposure_col = validate_scan_dataframe(df)

        # Use default AI channels if not specified
        if ai_channels is None:
            # Default to common channels
            # TODO: @tjferron - maybe we should make this configurable? or a larger set?
            ai_channels = ["Photodiode", "TEY signal", "AI 3 Izero"]

        # Build scan points
        points = []
        for idx, row in df.iterrows():
            motors = {col: float(row[col]) for col in motor_cols}

            if exposure_col:
                exposure = float(row[exposure_col])
            else:
                exposure = default_exposure

            point = ScanPoint(
                index=int(idx),
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

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def estimated_duration_seconds(
        self,
        motor_time: float = MOTOR_MOVE_TIME_PER_POINT,
        api_time: float = API_PROCESSING_TIME_PER_POINT,
    ) -> float:
        """
        Compute estimated scan duration.

        Per point: motor_time + api_time + exposure_time + delay_after_move
        (defaults: 0.1 s + 0.5 s + exposure + delay per point).

        Parameters
        ----------
        motor_time : float
            Time per point for motor movements (default 0.1 s).
        api_time : float
            Time per point for API processing (default 0.5 s).

        Returns
        -------
        float
            Total estimated duration in seconds.
        """
        total = 0.0
        for p in self.points:
            total += motor_time + api_time + p.exposure_time + p.delay_after_move
        return total

    def describe(
        self,
        motor_time: float = MOTOR_MOVE_TIME_PER_POINT,
        api_time: float = API_PROCESSING_TIME_PER_POINT,
    ) -> None:
        """
        Print a summary of the scan plan including unique values and estimated duration.

        Parameters
        ----------
        motor_time : float
            Time per point for motor movements (default 0.1 s).
        api_time : float
            Time per point for API processing (default 0.5 s).
        """
        lines: List[str] = []
        lines.append(f"Scan plan: {len(self.points)} points, {len(self.motor_names)} motors")
        lines.append("")
        lines.append("Unique values:")
        for m in self.motor_names:
            n = len({p.motors[m] for p in self.points})
            lines.append(f"  {m}: {n}")
        n_exp = len({p.exposure_time for p in self.points})
        lines.append(f"  exposure: {n_exp}")
        n_delay = len({p.delay_after_move for p in self.points})
        delay_val = self.points[0].delay_after_move if self.points else 0
        lines.append(f"  delay: {n_delay} ({delay_val} s)")
        lines.append("")
        duration = self.estimated_duration_seconds(motor_time=motor_time, api_time=api_time)
        minutes = duration / 60.0
        hours = duration / 3600.0
        if hours >= 1:
            time_str = f"{hours:.1f} h ({duration:.0f} s)"
        elif minutes >= 1:
            time_str = f"{minutes:.1f} min ({duration:.0f} s)"
        else:
            time_str = f"{duration:.1f} s"
        lines.append(f"Estimated duration: {time_str}")
        lines.append(f"  per point: {motor_time} s (motor) + {api_time} s (api) + exposure + {delay_val} s (delay)")
        print("\n".join(lines))


# ============================================================================
# Scan Executor
# ============================================================================


class ScanExecutor:
    """Executes scan plans with progress tracking and error recovery"""

    def __init__(self, server: BCSServer):
        self.server = server
        self.abort_flag = AbortFlag()
        self.current_scan: Optional[ScanPlan] = None

    async def abort(self):
        """Request abort of current scan"""
        await self.abort_flag.set()

    async def execute_point(
        self,
        point: ScanPoint,
        motor_timeout: float = 30.0,
        restore_motors: bool = False,
        use_shutter: bool = True,
    ) -> ScanResult:
        """
        Execute a single scan point with full error handling.

        Parameters:
            point: Scan point to execute
            motor_timeout: Timeout for motor moves
            restore_motors: Whether to restore motor positions after point
            use_shutter: If True, open/close shutter for this point; if False, assume shutter already open

        Returns:
            ScanResult with ufloat values

        Raises:
            MotorError, ShutterError, AcquisitionError, ScanAbortedError
        """
        # Check abort
        if await self.abort_flag.is_set():
            raise ScanAbortedError("Scan aborted before point execution")

        # Move motors (optionally restore on exit)
        async with motor_move(
            self.server,
            point.motors,
            timeout=motor_timeout,
            restore_on_exit=restore_motors,
        ) as initial_pos:  # noqa: F841
            # Wait for settling
            await wait_for_settle(point.delay_after_move, self.abort_flag)

            async def _acquire() -> dict:
                if await self.abort_flag.is_set():
                    raise ScanAbortedError("Scan aborted before acquisition")
                await self.server.acquire_data(
                    chans=point.ai_channels or self.current_scan.ai_channels,
                    time=point.exposure_time,
                )
                return await self.server.get_acquired_array(
                    chans=point.ai_channels or self.current_scan.ai_channels
                )

            if use_shutter:
                async with shutter_control(
                    self.server, shutter=self.current_scan.shutter, delay_before_open=0
                ):
                    result = await _acquire()
            else:
                result = await _acquire()

            ai_data = {}
            raw_data = {}
            for chan_data in result["chans"]:
                chan_name = chan_data["chan"]
                data = np.array(chan_data["data"], dtype=float)
                raw_data[chan_name] = data.tolist()

                if len(data) == 0:
                    ai_data[chan_name] = ufloat(np.nan, np.nan)
                else:
                    mean = np.nanmean(data)
                    std_err = np.nanstd(data, ddof=1) / np.sqrt(len(data))
                    ai_data[chan_name] = ufloat(mean, std_err)

        # Create result
        return ScanResult(
            index=point.index,
            motors=point.motors,
            ai_data=ai_data,
            exposure_time=point.exposure_time,
            timestamp=time.time(),
            raw_data=raw_data,
        )

    async def execute_scan(
        self, scan_plan: ScanPlan, progress: bool = True
    ) -> pd.DataFrame:
        """
        Execute complete scan plan.

        Parameters:
            scan_plan: Validated scan plan
            progress: Show progress bar (requires tqdm)

        Returns:
            DataFrame with mean and std columns for AI data

        Raises:
            Various scan errors
        """
        self.current_scan = scan_plan
        await self.abort_flag.clear()

        results = []

        # Create progress indicator
        if progress:
            try:
                from tqdm.asyncio import tqdm

                iterator = tqdm(scan_plan.points, desc="Scanning", unit="pt")
            except ImportError:
                print("tqdm not available, showing simple progress")
                iterator = scan_plan.points
                progress = False
        else:
            iterator = scan_plan.points

        async def _run_points() -> None:
            for i, point in enumerate(iterator):
                if not progress:
                    print(f"Point {i + 1}/{len(scan_plan.points)}", end="\r")
                result = await self.execute_point(
                    point, restore_motors=False, use_shutter=scan_plan.actuate_every
                )
                results.append(result)

        try:
            if scan_plan.actuate_every:
                await _run_points()
            else:
                async with shutter_control(
                    self.server, shutter=scan_plan.shutter, delay_before_open=0
                ):
                    await _run_points()

        except ScanAbortedError:
            print(f"\nScan aborted after {len(results)}/{len(scan_plan.points)} points")
            if not results:
                raise

        finally:
            self.current_scan = None
            if not progress:
                print()  # New line after progress

        # Convert to DataFrame
        df = pd.DataFrame([r.to_series() for r in results])
        return df

    