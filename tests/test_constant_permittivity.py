#!/usr/bin/env python
"""
Test 1: Constant Permittivity (Scalar Values)

Tests energy conservation for constant permittivity (scalar values) for:
- Single interface (zero layers)
- Single layer
- Both s and p polarizations
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from test_common_setup import (
    tolerance,
    eps_air_const,
    eps_silica_const,
    eps_Nb2O5_const,
)

print("=" * 80)
print("Test 1: Periodic Stack Air | [Nb2O5 | SiO2]10 | SiO2 @ 0°")
print("=" * 80)

wavelengths = np.linspace(1e-6, 2e-6, 300)

period_layers = []
for _ in range(10):
    period_layers.append(
        Layer(
            thickness=167e-9,
            optical_property={"type": "permittivity", "value": eps_Nb2O5_const},
        )
    )
    period_layers.append(
        Layer(
            thickness=257e-9,
            optical_property={"type": "permittivity", "value": eps_silica_const},
        )
    )

# Terminal SiO2 cap layer
period_layers.append(
    Layer(
        thickness=257e-9,
        optical_property={"type": "permittivity", "value": eps_silica_const},
    )
)

reflectance_spectra = {}

for pol in ["s", "p"]:
    multilayer = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=period_layers,
        eps_incident=eps_air_const,
        eps_exit=eps_silica_const,
    )
    R = multilayer.reflectance()
    T = multilayer.transmittance()
    reflectance_spectra[pol] = R
    R_plus_T = R + T
    max_dev = np.max(np.abs(R_plus_T - 1.0))
    passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {pol.upper()}-pol periodic stack: {status} | Max dev: {max_dev:.2e}")

print("\n" + "=" * 80)
print("Reflectance spectra (wavelength in nm):")
print("λ (nm)\tR_s\tR_p")
wavelength_nm = wavelengths * 1e9
R_s = reflectance_spectra.get("s")
R_p = reflectance_spectra.get("p")
if R_s is None or R_p is None:
    raise RuntimeError("Missing reflectance data for both polarizations.")

for w, rs, rp in zip(wavelength_nm, R_s, R_p):
    print(f"{w:8.2f}\t{rs:.6f}\t{rp:.6f}")
