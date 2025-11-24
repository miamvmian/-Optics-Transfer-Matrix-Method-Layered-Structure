#!/usr/bin/env python
"""
Test 7: Oblique Incidence with Wavelength-Dependent Permittivity

Tests energy conservation for oblique incidence (0° to 89°) with
wavelength-dependent permittivity at single interface.
"""

import numpy as np
from TMatrix import MultiLayerStructure
from test_common_setup import tolerance, wavelengths, eps_air_array, eps_glass_array

print("=" * 80)
print("Test 7: Oblique Incidence with Wavelength-Dependent Permittivity")
print("=" * 80)

angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 89.0]  # Include very high angles
results_angles = {"s": {"passed": 0, "failed": 0}, "p": {"passed": 0, "failed": 0}}

for angle in angles:
    for pol in ["s", "p"]:
        ml8 = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=angle,
            polarization=pol,
            layers=[],
            eps_incident=eps_air_array,
            eps_exit=eps_glass_array,
        )
        R8 = ml8.reflectance()
        T8 = ml8.transmittance()
        R_plus_T8 = R8 + T8
        max_dev8 = np.max(np.abs(R_plus_T8 - 1.0))
        passed8 = np.allclose(R_plus_T8, 1.0, atol=tolerance)

        if passed8:
            results_angles[pol]["passed"] += 1
        else:
            results_angles[pol]["failed"] += 1

        if not passed8 or angle in [0.0, 30.0, 60.0, 85.0]:
            status8 = "✓ PASS" if passed8 else "✗ FAIL"
            print(
                f"  {angle:3.0f}°, {pol.upper()}-pol: {status8} | Max dev: {max_dev8:.2e}"
            )

print(
    f"\n  Summary: s-pol passed: {results_angles['s']['passed']}, failed: {results_angles['s']['failed']}"
)
print(
    f"           p-pol passed: {results_angles['p']['passed']}, failed: {results_angles['p']['failed']}"
)
print("\n" + "=" * 80)

