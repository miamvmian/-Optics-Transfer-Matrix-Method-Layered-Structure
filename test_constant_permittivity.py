#!/usr/bin/env python
"""
Test 1: Constant Permittivity (Scalar Values)

Tests energy conservation for constant permittivity (scalar values) for:
- Single interface (zero layers)
- Single layer
- Both s and p polarizations
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from test_common_setup import tolerance, wavelengths, eps_air_const, eps_glass_const, eps_silica_const

print("=" * 80)
print("Test 1: Constant Permittivity (Scalar Values)")
print("=" * 80)

for pol in ["s", "p"]:
    # Single interface with constant permittivity
    ml1 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[],
        eps_incident=eps_air_const,
        eps_exit=eps_glass_const,
    )
    R1 = ml1.reflectance()
    T1 = ml1.transmittance()
    R_plus_T1 = R1 + T1
    max_dev1 = np.max(np.abs(R_plus_T1 - 1.0))
    passed1 = np.allclose(R_plus_T1, 1.0, atol=tolerance)

    # Single layer with constant permittivity
    layer_const = Layer(
        thickness=100e-9,
        optical_property={"type": "permittivity", "value": eps_silica_const},
    )
    ml2 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer_const],
        eps_incident=eps_air_const,
        eps_exit=eps_air_const,
    )
    R2 = ml2.reflectance()
    T2 = ml2.transmittance()
    R_plus_T2 = R2 + T2
    max_dev2 = np.max(np.abs(R_plus_T2 - 1.0))
    passed2 = np.allclose(R_plus_T2, 1.0, atol=tolerance)

    status1 = "✓ PASS" if passed1 else "✗ FAIL"
    status2 = "✓ PASS" if passed2 else "✗ FAIL"
    print(f"  {pol.upper()}-pol, single interface: {status1} | Max dev: {max_dev1:.2e}")
    print(f"  {pol.upper()}-pol, single layer: {status2} | Max dev: {max_dev2:.2e}")

print("\n" + "=" * 80)

