#!/usr/bin/env python
import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
"""
Test 3: Mixed Cases (Constant + Wavelength-Dependent)

Tests energy conservation for mixed cases where some materials have constant
permittivity and others have wavelength-dependent permittivity.
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from test_common_setup import (
    tolerance,
    wavelengths,
    eps_air_const,
    eps_glass_array,
    eps_silica_array,
)

print("=" * 80)
print("Test 3: Mixed Cases (Constant + Wavelength-Dependent)")
print("=" * 80)

for pol in ["s", "p"]:
    # Constant incident, wavelength-dependent exit
    ml5 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[],
        eps_incident=eps_air_const,
        eps_exit=eps_glass_array,
    )
    R5 = ml5.reflectance()
    T5 = ml5.transmittance()
    R_plus_T5 = R5 + T5
    max_dev5 = np.max(np.abs(R_plus_T5 - 1.0))
    passed5 = np.allclose(R_plus_T5, 1.0, atol=tolerance)

    # Wavelength-dependent layer, constant media
    layer_mixed = Layer(
        thickness=100e-9,
        optical_property={"type": "permittivity", "value": eps_silica_array},
    )
    ml6 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer_mixed],
        eps_incident=eps_air_const,
        eps_exit=eps_air_const,
    )
    R6 = ml6.reflectance()
    T6 = ml6.transmittance()
    R_plus_T6 = R6 + T6
    max_dev6 = np.max(np.abs(R_plus_T6 - 1.0))
    passed6 = np.allclose(R_plus_T6, 1.0, atol=tolerance)

    status5 = "✓ PASS" if passed5 else "✗ FAIL"
    status6 = "✓ PASS" if passed6 else "✗ FAIL"
    print(
        f"  {pol.upper()}-pol, const incident + array exit: {status5} | Max dev: {max_dev5:.2e}"
    )
    print(
        f"  {pol.upper()}-pol, const media + array layer: {status6} | Max dev: {max_dev6:.2e}"
    )

print("\n" + "=" * 80)

