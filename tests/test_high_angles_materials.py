#!/usr/bin/env python
import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
"""
Test 10: Very High Angles (up to 89°) with Multiple Material Combinations

Tests energy conservation at very high angles (up to 89°) with various material
combinations including Air, Glass, Silicon, Water, and reverse directions.
"""

import numpy as np
from TMatrix import MultiLayerStructure
from test_common_setup import (
    tolerance,
    wavelengths,
    eps_air_const,
    eps_glass_const,
    eps_silicon_const,
    eps_water_const,
    eps_air_array,
    eps_glass_array,
    eps_silicon_array,
    eps_water_array,
)

print("=" * 80)
print("Test 10: Very High Angles (up to 89°) with Multiple Material Combinations")
print("=" * 80)

# Test various material combinations at very high angles
high_angle_materials = [
    ("Air | Glass", eps_air_const, eps_glass_const, eps_air_array, eps_glass_array),
    (
        "Air | Silicon",
        eps_air_const,
        eps_silicon_const,
        eps_air_array,
        eps_silicon_array,
    ),
    ("Air | Water", eps_air_const, eps_water_const, eps_air_array, eps_water_array),
    (
        "Glass | Air",
        eps_glass_const,
        eps_air_const,
        eps_glass_array,
        eps_air_array,
    ),  # Reverse direction
]

high_angles = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 89.0]
results_high_angles = {"s": {"passed": 0, "failed": 0}, "p": {"passed": 0, "failed": 0}}

for (
    mat_name,
    eps_inc_const,
    eps_exit_const,
    eps_inc_array,
    eps_exit_array,
) in high_angle_materials:
    print(f"\n  Material combination: {mat_name}")
    for angle in high_angles:
        for pol in ["s", "p"]:
            # Test with constant permittivity
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=[],
                eps_incident=eps_inc_const,
                eps_exit=eps_exit_const,
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)

            if passed:
                results_high_angles[pol]["passed"] += 1
            else:
                results_high_angles[pol]["failed"] += 1

            # Print key cases
            if angle in [0.0, 30.0, 60.0, 85.0, 89.0]:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(
                    f"    {angle:3.0f}°, {pol.upper()}-pol (const): {status} | Max dev: {max_dev:.2e}"
                )

print(
    f"\n  Summary: s-pol passed: {results_high_angles['s']['passed']}, failed: {results_high_angles['s']['failed']}"
)
print(
    f"           p-pol passed: {results_high_angles['p']['passed']}, failed: {results_high_angles['p']['failed']}"
)
print("\n" + "=" * 80)

