#!/usr/bin/env python
import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
"""
Test 2: Wavelength-Dependent Permittivity (Arrays)

Tests energy conservation for wavelength-dependent permittivity (arrays) for:
- Single interface (zero layers)
- Single layer
- Both s and p polarizations
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from test_common_setup import (
    tolerance,
    wavelengths,
    n_wavelengths,
    eps_air_array,
    eps_glass_array,
    eps_silica_array,
)

print("=" * 80)
print("Test 2: Wavelength-Dependent Permittivity (Arrays)")
print("=" * 80)

for pol in ["s", "p"]:
    # Single interface with wavelength-dependent permittivity
    ml3 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[],
        eps_incident=eps_air_array,
        eps_exit=eps_glass_array,
    )
    R3 = ml3.reflectance()
    T3 = ml3.transmittance()
    R_plus_T3 = R3 + T3
    max_dev3 = np.max(np.abs(R_plus_T3 - 1.0))
    passed3 = np.allclose(R_plus_T3, 1.0, atol=tolerance)

    # Single layer with wavelength-dependent permittivity
    layer_array = Layer(
        thickness=100e-9,
        optical_property={"type": "permittivity", "value": eps_silica_array},
    )
    ml4 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer_array],
        eps_incident=eps_air_array,
        eps_exit=eps_air_array,
    )
    R4 = ml4.reflectance()
    T4 = ml4.transmittance()
    R_plus_T4 = R4 + T4
    max_dev4 = np.max(np.abs(R_plus_T4 - 1.0))
    passed4 = np.allclose(R_plus_T4, 1.0, atol=tolerance)

    status3 = "✓ PASS" if passed3 else "✗ FAIL"
    status4 = "✓ PASS" if passed4 else "✗ FAIL"
    print(f"  {pol.upper()}-pol, single interface: {status3} | Max dev: {max_dev3:.2e}")
    print(f"  {pol.upper()}-pol, single layer: {status4} | Max dev: {max_dev4:.2e}")

print("\n" + "=" * 80)

