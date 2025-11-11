#!/usr/bin/env python
"""
Comprehensive Energy Conservation Test for Same Incident/Exit Medium
Tests both normal and oblique incidence with varying numbers of layers
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure

print("=" * 80)
print("Energy Conservation Test: Same Incident/Exit Medium")
print("=" * 80)

tolerance = 1e-10
wavelengths = np.array([600e-9])
eps_air = 1.0

# Test angles
angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0]

# Test different numbers of layers
max_layers = 10

print(f"\nTesting from 0 to {max_layers} layers")
print(f"Angles: {', '.join([f'{a:.0f}°' for a in angles])}")
print(f"Incident/Exit medium: Air (eps = {eps_air})")
print("\n" + "=" * 80)

all_passed = True
failing_cases = []

for n_layers in range(0, max_layers + 1):
    # Create layers
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 5e-9)
        if i % 2 == 0:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 2.13}))
        else:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 5.76}))
    
    print(f"\n{n_layers} layer(s):")
    print("-" * 80)
    print(f"{'Angle':<8} {'s-pol R+T':<15} {'s-pol Status':<15} {'p-pol R+T':<15} {'p-pol Status':<15}")
    print("-" * 80)
    
    for angle in angles:
        results = {}
        for pol in ['s', 'p']:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=layers,
                eps_incident=eps_air,
                eps_exit=eps_air
            )
            R = ml.reflectance()[0]
            T = ml.transmittance()[0]
            R_plus_T = R + T
            max_dev = abs(R_plus_T - 1.0)
            passed = abs(R_plus_T - 1.0) < tolerance
            results[pol] = (R_plus_T, max_dev, passed)
            
            if not passed:
                all_passed = False
                failing_cases.append((n_layers, angle, pol, R_plus_T, max_dev))
        
        s_r_plus_t, s_dev, s_passed = results['s']
        p_r_plus_t, p_dev, p_passed = results['p']
        
        s_status = "✓ PASS" if s_passed else f"✗ FAIL ({s_dev:.2e})"
        p_status = "✓ PASS" if p_passed else f"✗ FAIL ({p_dev:.2e})"
        
        print(f"{angle:>6.0f}°   {s_r_plus_t:<15.10f} {s_status:<15} {p_r_plus_t:<15.10f} {p_status:<15}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

if all_passed:
    total_tests = (max_layers + 1) * len(angles) * 2  # layers * angles * polarizations
    print(f"\n✓ ALL TESTS PASSED!")
    print(f"  Total test cases: {total_tests}")
    print(f"  All cases satisfy energy conservation: R + T = 1.0")
    print(f"  Tested: 0 to {max_layers} layers, {len(angles)} angles, both s and p polarizations")
    print(f"  Tolerance: {tolerance}")
else:
    print(f"\n✗ SOME TESTS FAILED")
    print(f"  Total failing cases: {len(failing_cases)}")
    print("\n  Failing cases:")
    print(f"  {'Layers':<8} {'Angle':<8} {'Pol':<6} {'R+T':<15} {'Deviation':<15}")
    print("  " + "-" * 60)
    for n_layers, angle, pol, r_plus_t, dev in failing_cases:
        print(f"  {n_layers:<8} {angle:>6.0f}°   {pol:<6} {r_plus_t:<15.10f} {dev:<15.2e}")

print("\n" + "=" * 80)

# Additional verification: Check a few specific cases in detail
print("\nDetailed Verification (Sample Cases):")
print("=" * 80)

sample_cases = [
    (0, 0.0, "Normal, single interface"),
    (0, 30.0, "Oblique 30°, single interface"),
    (1, 0.0, "Normal, 1 layer"),
    (1, 30.0, "Oblique 30°, 1 layer"),
    (5, 0.0, "Normal, 5 layers"),
    (5, 45.0, "Oblique 45°, 5 layers"),
    (10, 0.0, "Normal, 10 layers"),
    (10, 60.0, "Oblique 60°, 10 layers"),
]

for n_layers, angle, desc in sample_cases:
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 5e-9)
        if i % 2 == 0:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 2.13}))
        else:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 5.76}))
    
    print(f"\n{desc}:")
    for pol in ['s', 'p']:
        ml = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=angle,
            polarization=pol,
            layers=layers,
            eps_incident=eps_air,
            eps_exit=eps_air
        )
        R = ml.reflectance()[0]
        T = ml.transmittance()[0]
        R_plus_T = R + T
        dev = abs(R_plus_T - 1.0)
        print(f"  {pol}-pol: R = {R:.10f}, T = {T:.10f}, R+T = {R_plus_T:.10f}, deviation = {dev:.2e}")

print("\n" + "=" * 80)

