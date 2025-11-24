#!/usr/bin/env python
import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
"""
Test 11: Layer Thickness Variations

Tests energy conservation for single layers with different thicknesses
(50 nm, 100 nm, 200 nm, 500 nm) at various angles.
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from test_common_setup import tolerance, wavelengths, eps_air_const, eps_glass_const

print("=" * 80)
print("Test 11: Layer Thickness Variations")
print("=" * 80)

layer_thicknesses = [50e-9, 100e-9, 200e-9, 500e-9]  # Different thicknesses
test_angles_thickness = [0.0, 30.0, 45.0, 60.0]
results_thickness = {"s": {"passed": 0, "failed": 0}, "p": {"passed": 0, "failed": 0}}

for thickness in layer_thicknesses:
    print(f"\n  Layer thickness: {thickness*1e9:.0f} nm")
    layer = Layer(
        thickness=thickness,
        optical_property={"type": "permittivity", "value": eps_glass_const},
    )

    for angle in test_angles_thickness:
        for pol in ["s", "p"]:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=[layer],
                eps_incident=eps_air_const,
                eps_exit=eps_air_const,
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)

            if passed:
                results_thickness[pol]["passed"] += 1
            else:
                results_thickness[pol]["failed"] += 1

            # Print key cases
            if angle in [0.0, 30.0, 60.0]:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(
                    f"    {angle:3.0f}°, {pol.upper()}-pol: {status} | Max dev: {max_dev:.2e}"
                )

print(
    f"\n  Summary: s-pol passed: {results_thickness['s']['passed']}, failed: {results_thickness['s']['failed']}"
)
print(
    f"           p-pol passed: {results_thickness['p']['passed']}, failed: {results_thickness['p']['failed']}"
)
print("\n" + "=" * 80)

