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
    E_in,
)

# Set input electric field
# E_in is imported from test_common_setup: E_in = P_in / Z_air_const
# where P_in = 1.0 W and Z_air_const is the impedance of air

# Open output file for writing results
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)  # Create data directory if it doesn't exist
output_file = data_dir / "tmatrix_30DEG.txt"
with open(output_file, "w") as f:
    header = "=" * 80 + "\n"
    header += "Test 1: Periodic Stack Air | [Nb2O5 | SiO2]10 | SiO2 @ 30°\n"
    header += f"Input electric field: E_in = {E_in:.6e} V/m\n"
    header += "=" * 80 + "\n"
    print(header, end="")
    f.write(header)

    wavelengths = np.linspace(1e-6, 2e-6, 200)

    period_layers = []
    for _ in range(10):
        period_layers.append(
            Layer(
                thickness=159e-9,
                optical_property={"type": "permittivity", "value": eps_Nb2O5_const},
            )
        )
        period_layers.append(
            Layer(
                thickness=246e-9,
                optical_property={"type": "permittivity", "value": eps_silica_const},
            )
        )

    # Terminal SiO2 cap layer
    period_layers.append(
        Layer(
            thickness=246e-9,
            optical_property={"type": "permittivity", "value": eps_silica_const},
        )
    )

    reflectance_spectra = {}

    for pol in ["s", "p"]:
        multilayer = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=30.0,
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
        line = (
            f"  {pol.upper()}-pol periodic stack: {status} | Max dev: {max_dev:.2e}\n"
        )
        print(line, end="")
        f.write(line)

    table_header = "\n" + "=" * 80 + "\n"
    table_header += "Reflectance spectra (wavelength in nm):\n"
    table_header += "λ (nm)\tR_s\tR_p\n"
    print(table_header, end="")
    f.write(table_header)

    wavelength_nm = wavelengths * 1e9
    R_s = reflectance_spectra.get("s")
    R_p = reflectance_spectra.get("p")
    if R_s is None or R_p is None:
        raise RuntimeError("Missing reflectance data for both polarizations.")

    for w, rs, rp in zip(wavelength_nm, R_s, R_p):
        line = f"{w:8.2f}\t{rs:.6f}\t{rp:.6f}\n"
        print(line, end="")
        f.write(line)
