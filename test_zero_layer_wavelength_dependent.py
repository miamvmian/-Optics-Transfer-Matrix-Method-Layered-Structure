#!/usr/bin/env python
"""
Test 5: Zero Layers (Single Interface) with Wavelength-Dependent Permittivity

Tests energy conservation for single interface (zero layers) with various
configurations of wavelength-dependent permittivity at multiple angles.
"""

import numpy as np
from TMatrix import MultiLayerStructure
from test_common_setup import (
    tolerance,
    wavelengths,
    eps_air_const,
    eps_glass_const,
    eps_air_array,
    eps_glass_array,
)

print("=" * 80)
print("Test 5: Zero Layers (Single Interface) with Wavelength-Dependent Permittivity")
print("=" * 80)

# Test various configurations of single interface with wavelength-dependent permittivity
test_configs = [
    ("Both media wavelength-dependent", eps_air_array, eps_glass_array),
    ("Incident wavelength-dependent, exit constant", eps_air_array, eps_glass_const),
    ("Incident constant, exit wavelength-dependent", eps_air_const, eps_glass_array),
    ("Both constant (baseline)", eps_air_const, eps_glass_const),
]

angles_interface = [
    0.0,
    15.0,
    30.0,
    45.0,
    60.0,
    75.0,
    85.0,
    89.0,
]  # Include very high angles
results_interface = {"s": {"passed": 0, "failed": 0}, "p": {"passed": 0, "failed": 0}}

for config_name, eps_inc, eps_exit in test_configs:
    print(f"\n  Configuration: {config_name}")
    for angle in angles_interface:
        for pol in ["s", "p"]:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=[],  # Zero layers = single interface
                eps_incident=eps_inc,
                eps_exit=eps_exit,
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)

            if passed:
                results_interface[pol]["passed"] += 1
            else:
                results_interface[pol]["failed"] += 1

            # Print key cases
            if angle in [0.0, 30.0, 45.0, 60.0, 85.0]:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(
                    f"    {angle:3.0f}°, {pol.upper()}-pol: {status} | Max dev: {max_dev:.2e}"
                )

print(
    f"\n  Summary: s-pol passed: {results_interface['s']['passed']}, failed: {results_interface['s']['failed']}"
)
print(
    f"           p-pol passed: {results_interface['p']['passed']}, failed: {results_interface['p']['failed']}"
)
print("\n" + "=" * 80)

