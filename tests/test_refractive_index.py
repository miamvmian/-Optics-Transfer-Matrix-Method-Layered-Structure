#!/usr/bin/env python
import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
"""
Test 9: Refractive Index Type (Converted to Permittivity)

Tests energy conservation when materials are specified using refractive index
instead of permittivity. The refractive index should be converted to permittivity
(eps = n^2).
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from test_common_setup import (
    tolerance,
    wavelengths,
    eps_air_const,
    eps_air_array,
    n_glass_const,
    n_glass_array,
)

print("=" * 80)
print("Test 9: Refractive Index Type (Converted to Permittivity)")
print("=" * 80)

# Test with constant refractive index
for pol in ["s", "p"]:
    layer_n = Layer(
        thickness=100e-9,
        optical_property={"type": "refractive_index", "value": n_glass_const},
    )
    ml9 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer_n],
        eps_incident=eps_air_const,
        eps_exit=eps_air_const,
    )
    R9 = ml9.reflectance()
    T9 = ml9.transmittance()
    R_plus_T9 = R9 + T9
    max_dev9 = np.max(np.abs(R_plus_T9 - 1.0))
    passed9 = np.allclose(R_plus_T9, 1.0, atol=tolerance)

    status9 = "✓ PASS" if passed9 else "✗ FAIL"
    print(
        f"  {pol.upper()}-pol, refractive_index (const): {status9} | Max dev: {max_dev9:.2e}"
    )

# Test with wavelength-dependent refractive index
for pol in ["s", "p"]:
    layer_n_array = Layer(
        thickness=100e-9,
        optical_property={"type": "refractive_index", "value": n_glass_array},
    )
    ml10 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer_n_array],
        eps_incident=eps_air_array,
        eps_exit=eps_air_array,
    )
    R10 = ml10.reflectance()
    T10 = ml10.transmittance()
    R_plus_T10 = R10 + T10
    max_dev10 = np.max(np.abs(R_plus_T10 - 1.0))
    passed10 = np.allclose(R_plus_T10, 1.0, atol=tolerance)

    status10 = "✓ PASS" if passed10 else "✗ FAIL"
    print(
        f"  {pol.upper()}-pol, refractive_index (array): {status10} | Max dev: {max_dev10:.2e}"
    )

print("\n" + "=" * 80)

