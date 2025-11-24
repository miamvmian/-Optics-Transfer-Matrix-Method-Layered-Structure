#!/usr/bin/env python
import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
"""
Test 13: Same Incident/Exit Medium (Comprehensive)

Tests energy conservation for cases where the incident and exit media are the
same, with 0-10 layers at various angles.
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from test_common_setup import (
    tolerance,
    wavelengths,
    eps_air_const,
    eps_silica_const,
    eps_titanium_const,
)

print("=" * 80)
print("Test 13: Same Incident/Exit Medium (Comprehensive)")
print("=" * 80)

same_medium_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0]
results_same_medium = {"s": {"passed": 0, "failed": 0}, "p": {"passed": 0, "failed": 0}}

for n_layers in range(0, 11):
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 5e-9)
        if i % 2 == 0:
            layers.append(
                Layer(
                    thickness=thickness,
                    optical_property={
                        "type": "permittivity",
                        "value": eps_silica_const,
                    },
                )
            )
        else:
            layers.append(
                Layer(
                    thickness=thickness,
                    optical_property={
                        "type": "permittivity",
                        "value": eps_titanium_const,
                    },
                )
            )

    for angle in same_medium_angles:
        for pol in ["s", "p"]:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=layers,
                eps_incident=eps_air_const,
                eps_exit=eps_air_const,
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)

            if passed:
                results_same_medium[pol]["passed"] += 1
            else:
                results_same_medium[pol]["failed"] += 1

            # Print only failures or key cases
            if not passed or (n_layers in [0, 1, 5, 10] and angle in [0.0, 45.0, 85.0]):
                status = "✓ PASS" if passed else "✗ FAIL"
                print(
                    f"  {n_layers} layers, {angle:3.0f}°, {pol.upper()}-pol: {status} | Max dev: {max_dev:.2e}"
                )

print(
    f"\n  Summary: s-pol passed: {results_same_medium['s']['passed']}, failed: {results_same_medium['s']['failed']}"
)
print(
    f"           p-pol passed: {results_same_medium['p']['passed']}, failed: {results_same_medium['p']['failed']}"
)
print("\n" + "=" * 80)

