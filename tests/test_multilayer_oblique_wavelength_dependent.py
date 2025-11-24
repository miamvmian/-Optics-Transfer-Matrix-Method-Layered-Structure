#!/usr/bin/env python
import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
"""
Test 8: Multiple Layers at Different Angles with Wavelength-Dependent Permittivity

Tests energy conservation for multiple layers (0, 1, 2, 3, 5, 10) at different
angles with wavelength-dependent permittivity.
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from test_common_setup import (
    tolerance,
    wavelengths,
    eps_air_array,
    eps_glass_array,
    eps_silica_array,
    eps_titanium_array,
)

print("=" * 80)
print(
    "Test 8: Multiple Layers at Different Angles with Wavelength-Dependent Permittivity"
)
print("=" * 80)

layer_counts = [0, 1, 2, 3, 5, 10]
test_angles = [0.0, 30.0, 45.0, 60.0]
results_combined = {"s": {"passed": 0, "failed": 0}, "p": {"passed": 0, "failed": 0}}

for n_layers in layer_counts:
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 5e-9)
        if i % 2 == 0:
            layers.append(
                Layer(
                    thickness=thickness,
                    optical_property={
                        "type": "permittivity",
                        "value": eps_silica_array,
                    },
                )
            )
        else:
            layers.append(
                Layer(
                    thickness=thickness,
                    optical_property={
                        "type": "permittivity",
                        "value": eps_titanium_array,
                    },
                )
            )

    for angle in test_angles:
        for pol in ["s", "p"]:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=layers,
                eps_incident=eps_air_array,
                eps_exit=eps_glass_array,
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)

            if passed:
                results_combined[pol]["passed"] += 1
            else:
                results_combined[pol]["failed"] += 1

            # Print only failures or key cases
            if not passed or (n_layers in [0, 1, 5] and angle in [0.0, 45.0]):
                status = "✓ PASS" if passed else "✗ FAIL"
                print(
                    f"  {n_layers} layers, {angle:3.0f}°, {pol.upper()}-pol: {status} | Max dev: {max_dev:.2e}"
                )

print(
    f"\n  Summary: s-pol passed: {results_combined['s']['passed']}, failed: {results_combined['s']['failed']}"
)
print(
    f"           p-pol passed: {results_combined['p']['passed']}, failed: {results_combined['p']['failed']}"
)
print("\n" + "=" * 80)

