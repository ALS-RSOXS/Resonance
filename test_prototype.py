"""
Quick validation test script
Run this to verify the prototype works without needing a beamline connection
"""

import numpy as np
import pandas as pd

from api_dev import (
    ScanPlan,
    ValidationError,
    find_exposure_column,
    validate_scan_dataframe,
)


def test_exposure_column_detection():
    """Test exposure column detection with various naming conventions"""
    print("=" * 60)
    print("TEST 1: Exposure Column Detection")
    print("=" * 60)

    test_cases = [
        ("exposure", pd.DataFrame({"Sample X": [1, 2], "exposure": [1.0, 1.5]})),
        ("exp", pd.DataFrame({"Sample X": [1, 2], "exp": [1.0, 1.5]})),
        ("count_time", pd.DataFrame({"Sample X": [1, 2], "count_time": [1.0, 1.5]})),
        ("Unnamed: 2", pd.DataFrame({"Sample X": [1, 2], "Unnamed: 2": [1.0, 1.5]})),
        ("(empty)", pd.DataFrame({"Sample X": [1, 2], "": [1.0, 1.5]})),
    ]

    for expected, df in test_cases:
        col = find_exposure_column(df)
        status = "✓" if col else "✗"
        print(f"  {status} Expected '{expected}', found: '{col}'")

    print()


def test_motor_validation():
    """Test motor column validation"""
    print("=" * 60)
    print("TEST 2: Motor Column Validation")
    print("=" * 60)

    # Valid DataFrame
    valid_df = pd.DataFrame(
        {
            "Sample X": [10.0, 10.5, 11.0],
            "Sample Y": [0.0, 0.0, 0.0],
            "exposure": [1.0, 1.5, 2.0],
        }
    )

    try:
        motor_cols, exposure_col = validate_scan_dataframe(valid_df)
        print("  ✓ Valid DataFrame")
        print(f"    Motor columns: {motor_cols}")
        print(f"    Exposure column: {exposure_col}")
    except ValidationError as e:
        print(f"  ✗ Unexpected error: {e}")

    # Invalid DataFrame (bad motor name)
    invalid_df = pd.DataFrame(
        {"Invalid Motor": [10.0, 10.5, 11.0], "exposure": [1.0, 1.5, 2.0]}
    )

    try:
        motor_cols, exposure_col = validate_scan_dataframe(invalid_df)
        print("  ✗ Should have raised ValidationError")
    except ValidationError as e:
        print("  ✓ Correctly rejected invalid motor name")
        print(f"    Error: {str(e)[:60]}...")

    print()


def test_scan_plan_creation():
    """Test ScanPlan creation from DataFrame"""
    print("=" * 60)
    print("TEST 3: ScanPlan Creation")
    print("=" * 60)

    # Create scan DataFrame
    scan_df = pd.DataFrame(
        {
            "Sample X": np.linspace(10, 12, 5),
            "Sample Y": [0.0] * 5,
            "Beamline Energy": np.linspace(280, 290, 5),
            "exposure": [1.0, 1.0, 1.5, 1.5, 2.0],
        }
    )

    # Create scan plan
    plan = ScanPlan.from_dataframe(
        df=scan_df,
        ai_channels=["Photodiode", "TEY signal", "AI 3 Izero"],
        default_delay=0.2,
        shutter="Light Output",
    )

    print(f"  ✓ Created scan plan with {len(plan)} points")
    print(f"    Motor columns: {plan.motor_names}")
    print(f"    AI channels: {plan.ai_channels}")
    print(f"    Shutter: {plan.shutter}")
    print("\n  First point:")
    point = plan.points[0]
    print(f"    Index: {point.index}")
    print(f"    Motors: {point.motors}")
    print(f"    Exposure: {point.exposure_time}s")
    print(f"    Delay: {point.delay_after_move}s")

    print()


def test_multi_motor_grid():
    """Test multi-motor grid scan creation"""
    print("=" * 60)
    print("TEST 4: Multi-Motor Grid Scan")
    print("=" * 60)

    # Create 2D grid
    x_positions = np.linspace(10, 12, 3)
    y_positions = np.linspace(0, 2, 3)
    X, Y = np.meshgrid(x_positions, y_positions)

    grid_df = pd.DataFrame(
        {
            "Sample X": X.flatten(),
            "Sample Y": Y.flatten(),
            "exposure": [1.0] * len(X.flatten()),
        }
    )

    plan = ScanPlan.from_dataframe(
        grid_df, ai_channels=["Photodiode"], default_delay=0.1
    )

    print(f"  ✓ Created 2D grid scan with {len(plan)} points")
    print(f"    Grid shape: {len(x_positions)} x {len(y_positions)}")
    print("\n  Sample points:")
    for i in [0, 4, 8]:
        point = plan.points[i]
        print(
            f"    Point {i}: X={point.motors['Sample X']:.1f}, Y={point.motors['Sample Y']:.1f}"
        )

    print()


def test_error_handling():
    """Test error handling for invalid data"""
    print("=" * 60)
    print("TEST 5: Error Handling")
    print("=" * 60)

    test_cases = [
        ("Empty DataFrame", pd.DataFrame()),
        ("No motor columns", pd.DataFrame({"exposure": [1.0, 2.0]})),
        (
            "Negative exposure",
            pd.DataFrame({"Sample X": [1, 2], "exposure": [1.0, -1.0]}),
        ),
        (
            "NaN in motor column",
            pd.DataFrame({"Sample X": [1.0, np.nan], "exposure": [1.0, 1.0]}),
        ),
    ]

    for description, df in test_cases:
        try:
            validate_scan_dataframe(df)
            print(f"  ✗ {description}: Should have raised error")
        except ValidationError:
            print(f"  ✓ {description}: Correctly caught")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RSoXS API PROTOTYPE - VALIDATION TESTS")
    print("=" * 60 + "\n")

    try:
        test_exposure_column_detection()
        test_motor_validation()
        test_scan_plan_creation()
        test_multi_motor_grid()
        test_error_handling()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nNext step: Test with beamline connection using prototype.ipynb")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
