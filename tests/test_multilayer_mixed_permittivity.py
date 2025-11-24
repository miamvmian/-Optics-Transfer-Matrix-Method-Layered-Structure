#!/usr/bin/env python
import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
"""
Test 4: Multiple Layers with Mixed Permittivity Types

Tests energy conservation for multiple layers where some layers have constant
permittivity and others have wavelength-dependent permittivity.
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from test_common_setup import (
    tolerance,
    wavelengths,
    eps_air_const,
    eps_silica_const,
    eps_titanium_array,
)

print("=" * 80)
print("Test 4: Multiple Layers with Mixed Permittivity Types")
print("=" * 80)

for pol in ["s", "p"]:
    # Two layers: one constant, one wavelength-dependent
    layer1_const = Layer(
        thickness=100e-9,
        optical_property={"type": "permittivity", "value": eps_silica_const},
    )
    layer2_array = Layer(
        thickness=150e-9,
        optical_property={"type": "permittivity", "value": eps_titanium_array},
    )
    ml7 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer1_const, layer2_array],
        eps_incident=eps_air_const,
        eps_exit=eps_air_const,
    )
    R7 = ml7.reflectance()
    T7 = ml7.transmittance()
    R_plus_T7 = R7 + T7
    max_dev7 = np.max(np.abs(R_plus_T7 - 1.0))
    passed7 = np.allclose(R_plus_T7, 1.0, atol=tolerance)

    status7 = "✓ PASS" if passed7 else "✗ FAIL"
    print(
        f"  {pol.upper()}-pol, 2 layers (const + array): {status7} | Max dev: {max_dev7:.2e}"
    )

print("\n" + "=" * 80)

