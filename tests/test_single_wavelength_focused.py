#!/usr/bin/env python
import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
"""
Test 12: Single Wavelength Focused Tests

Tests energy conservation for focused test cases using a single wavelength
(600 nm) covering various scenarios with pattern analysis.
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from test_common_setup import (
    tolerance,
    wavelength_single,
    eps_air_const,
    eps_glass_const,
    eps_silica_const,
    eps_titanium_const,
)

print("=" * 80)
print("Test 12: Single Wavelength Focused Tests")
print("=" * 80)

# Focused test cases with single wavelength
focused_cases = [
    ("0 layers, normal, same exit", 0, 0.0, eps_air_const, eps_air_const),
    ("0 layers, normal, glass exit", 0, 0.0, eps_air_const, eps_glass_const),
    ("0 layers, oblique 30°, same exit", 0, 30.0, eps_air_const, eps_air_const),
    ("0 layers, oblique 30°, glass exit", 0, 30.0, eps_air_const, eps_glass_const),
    ("1 layer, normal, same exit", 1, 0.0, eps_air_const, eps_air_const),
    ("1 layer, normal, glass exit", 1, 0.0, eps_air_const, eps_glass_const),
    ("2 layers, normal, same exit", 2, 0.0, eps_air_const, eps_air_const),
    ("2 layers, normal, glass exit", 2, 0.0, eps_air_const, eps_glass_const),
    ("5 layers, oblique 45°, same exit", 5, 45.0, eps_air_const, eps_air_const),
    ("10 layers, oblique 60°, glass exit", 10, 60.0, eps_air_const, eps_glass_const),
]

results_focused = []
for desc, n_layers, angle, eps_inc, eps_exit in focused_cases:
    layers = []
    for i in range(n_layers):
        thickness = 100e-9
        if i % 2 == 0:
            layers.append(
                Layer(
                    thickness=thickness,
                    optical_property={
                        "type": "permittivity",
                        "value": eps_silica_const,
                    },
                )
            )
        else:
            layers.append(
                Layer(
                    thickness=thickness,
                    optical_property={
                        "type": "permittivity",
                        "value": eps_titanium_const,
                    },
                )
            )

    for pol in ["s", "p"]:
        ml = MultiLayerStructure(
            wavelengths=wavelength_single,
            angle_degrees=angle,
            polarization=pol,
            layers=layers,
            eps_incident=eps_inc,
            eps_exit=eps_exit,
        )
        R = ml.reflectance()[0]
        T = ml.transmittance()[0]
        R_plus_T = R + T
        max_dev = abs(R_plus_T - 1.0)
        passed = abs(R_plus_T - 1.0) < tolerance
        results_focused.append((desc, pol, n_layers, angle, R_plus_T, max_dev, passed))

        if not passed or (n_layers <= 2 and angle <= 30.0):
            status = "✓ PASS" if passed else "✗ FAIL"
            print(
                f"  {desc}, {pol.upper()}-pol: {status} | R+T = {R_plus_T:.10f} | Dev: {max_dev:.2e}"
            )

# Pattern analysis for focused tests
failing_focused = [r for r in results_focused if not r[6]]
if failing_focused:
    print(f"\n  Pattern Analysis: {len(failing_focused)} failing cases")
    by_layers = {}
    for desc, pol, n_layers, angle, r_plus_t, dev, passed in failing_focused:
        if n_layers not in by_layers:
            by_layers[n_layers] = []
        by_layers[n_layers].append((angle, r_plus_t, dev))
    for n_layers in sorted(by_layers.keys()):
        cases = by_layers[n_layers]
        avg_r_plus_t = np.mean([r_plus_t for _, r_plus_t, _ in cases])
        print(
            f"    {n_layers} layer(s): {len(cases)} cases, avg R+T = {avg_r_plus_t:.10f}"
        )

print("\n" + "=" * 80)

