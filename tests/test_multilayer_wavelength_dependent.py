#!/usr/bin/env python
import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
"""
Test 6: Different Numbers of Layers with Wavelength-Dependent Permittivity

Tests energy conservation for different numbers of layers (0 to 10) with
wavelength-dependent permittivity at various angles.
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from test_common_setup import (
    tolerance,
    wavelengths,
    eps_air_array,
    eps_glass_array,
    eps_silica_array,
    eps_titanium_const,
)

print("=" * 80)
print("Test 6: Different Numbers of Layers with Wavelength-Dependent Permittivity")
print("=" * 80)

max_layers = 10
angles_test6 = [0.0, 30.0, 45.0]
results_layers = {"s": {"passed": 0, "failed": 0}, "p": {"passed": 0, "failed": 0}}

for n_layers in range(0, max_layers + 1):
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 5e-9)
        if i % 2 == 0:
            # Even layers: wavelength-dependent permittivity
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
            # Odd layers: constant permittivity
            layers.append(
                Layer(
                    thickness=thickness,
                    optical_property={
                        "type": "permittivity",
                        "value": eps_titanium_const,
                    },
                )
            )

    for angle in angles_test6:
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
                results_layers[pol]["passed"] += 1
            else:
                results_layers[pol]["failed"] += 1

            # Print only for key cases to avoid too much output
            if (n_layers <= 2 or n_layers == 5 or n_layers == 10) and angle in [
                0.0,
                45.0,
            ]:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(
                    f"  {n_layers} layers, {angle:3.0f}°, {pol.upper()}-pol: {status} | Max dev: {max_dev:.2e}"
                )

print(
    f"\n  Summary: s-pol passed: {results_layers['s']['passed']}, failed: {results_layers['s']['failed']}"
)
print(
    f"           p-pol passed: {results_layers['p']['passed']}, failed: {results_layers['p']['failed']}"
)
print("\n" + "=" * 80)

