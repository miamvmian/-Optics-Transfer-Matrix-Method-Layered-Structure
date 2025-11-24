#!/usr/bin/env python
"""
Test 14: Pattern Analysis - Different Exit Medium

Tests energy conservation with different exit medium and provides detailed
pattern analysis grouped by number of layers and polarization.
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
print("Test 14: Pattern Analysis - Different Exit Medium")
print("=" * 80)

# Test with different exit medium and analyze patterns
pattern_angles = [15.0, 30.0, 45.0, 60.0, 75.0, 85.0]
pattern_results = []

for n_layers in range(0, 11):
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 5e-9)
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

    for angle in pattern_angles:
        for pol in ["s", "p"]:
            ml = MultiLayerStructure(
                wavelengths=wavelength_single,
                angle_degrees=angle,
                polarization=pol,
                layers=layers,
                eps_incident=eps_air_const,
                eps_exit=eps_glass_const,
            )
            R = ml.reflectance()[0]
            T = ml.transmittance()[0]
            R_plus_T = R + T
            max_dev = abs(R_plus_T - 1.0)
            passed = abs(R_plus_T - 1.0) < tolerance
            pattern_results.append((n_layers, angle, pol, R_plus_T, max_dev, passed))

# Pattern analysis: group by layers
by_layers_pattern = {}
for n_layers, angle, pol, r_plus_t, dev, passed in pattern_results:
    if n_layers not in by_layers_pattern:
        by_layers_pattern[n_layers] = {"s": [], "p": []}
    by_layers_pattern[n_layers][pol].append((angle, r_plus_t, dev, passed))

print("\n  Pattern Analysis by Number of Layers:")
print(
    f"  {'Layers':<8} {'Polarization':<12} {'Cases':<8} {'Passed':<8} {'Failed':<8} {'Avg R+T':<15}"
)
print("  " + "-" * 70)

for n_layers in sorted(by_layers_pattern.keys()):
    for pol in ["s", "p"]:
        cases = by_layers_pattern[n_layers][pol]
        passed_count = sum(1 for _, _, _, p in cases if p)
        failed_count = len(cases) - passed_count
        if cases:
            avg_r_plus_t = np.mean([r_plus_t for _, r_plus_t, _, _ in cases])
            print(
                f"  {n_layers:<8} {pol.upper():<12} {len(cases):<8} {passed_count:<8} {failed_count:<8} {avg_r_plus_t:<15.10f}"
            )

print("\n" + "=" * 80)

