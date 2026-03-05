"""
RSoXS Beamline Control API

Provides the Beamline interface and typed accessors for the ALS RSoXS beamline,
built on BCSz for hardware control. Core components:

- Beamline: high-level facade (scan_from_dataframe, abort_scan, is_scanning)
- MotorAccessor, AIAccessor, DIOAccessor: typed hardware accessors
- ScanPlan, ScanExecutor: DataFrame-driven scan orchestration
- Primitives: AbortFlag, motor_move, shutter_control, wait_for_motors, wait_for_settle
- Types: Motor, AI, DIO, Command literals and all domain exceptions
"""

__version__ = "0.1.0"


from resonance.api.core import (
    AbortFlag,
    AIAccessor,
    Beamline,
    Connection,
    DIOAccessor,
    MotorAccessor,
    MotorState,
    ScanExecutor,
    ScanPlan,
    motor_move,
    shutter_control,
    wait_for_motors,
    wait_for_settle,
)
from resonance.api.header_map import HEADER_NAMES, normalize_header
from resonance.api.types import (
    # Type literals
    AI,
    DIO,
    AcquisitionError,
    Command,
    Motor,
    MotorError,
    MotorTimeoutError,
    RsoxsError,
    ScanAbortedError,
    ScanPoint,
    ScanResult,
    ShutterError,
    ValidationError,
    ai,
    command,
    dio,
    motor,
)

# Utility functions
from resonance.api.utils import (
    calculate_center_of_mass,
    create_energy_scan,
    create_grid_scan,
    create_line_scan,
    find_peak_position,
    merge_scans,
    resample_scan_data,
)

# Validation utilities
from resonance.api.validation import (
    find_exposure_column,
    validate_motor_columns,
    validate_scan_dataframe,
)

__all__ = [
    # Version
    "__version__",
    # Header map
    "HEADER_NAMES",
    "normalize_header",
    # Exceptions
    "RsoxsError",
    "MotorError",
    "MotorTimeoutError",
    "ShutterError",
    "AcquisitionError",
    "ValidationError",
    "ScanAbortedError",
    # Data structures
    "ScanPoint",
    "ScanResult",
    # Types
    "AI",
    "Motor",
    "DIO",
    "Command",
    "ai",
    "motor",
    "dio",
    "command",
    # Validation
    "find_exposure_column",
    "validate_motor_columns",
    "validate_scan_dataframe",
    # Core primitives and accessors
    "AbortFlag",
    "wait_for_motors",
    "wait_for_settle",
    "motor_move",
    "shutter_control",
    "AIAccessor",
    "Beamline",
    "Connection",
    "DIOAccessor",
    "MotorAccessor",
    "MotorState",
    "ScanExecutor",
    "ScanPlan",
    # Utilities
    "create_grid_scan",
    "create_line_scan",
    "create_energy_scan",
    "find_peak_position",
    "calculate_center_of_mass",
    "resample_scan_data",
    "merge_scans",
]
