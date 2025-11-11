#!/usr/bin/env python
"""
Energy Conservation Test: P-polarization, Oblique Incidence, Different Exit Medium
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure

print("=" * 80)
print("Energy Conservation Test: P-polarization, Oblique Incidence")
print("Different Incident/Exit Media")
print("=" * 80)

tolerance = 1e-10
wavelengths = np.array([600e-9])
eps_air = 1.0
eps_glass = 2.25

# Test angles (oblique incidence only, excluding 0°)
angles = [15.0, 30.0, 45.0, 60.0, 75.0, 85.0]

# Test different numbers of layers
max_layers = 10

print(f"\nTesting from 0 to {max_layers} layers")
print(f"Angles: {', '.join([f'{a:.0f}°' for a in angles])}")
print(f"Polarization: p (TM)")
print(f"Incident medium: Air (eps = {eps_air})")
print(f"Exit medium: Glass (eps = {eps_glass})")
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
    print(f"{'Angle':<8} {'R':<15} {'T':<15} {'R+T':<15} {'Status':<15} {'Deviation':<15}")
    print("-" * 80)
    
    for angle in angles:
        ml = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=angle,
            polarization='p',
            layers=layers,
            eps_incident=eps_air,
            eps_exit=eps_glass
        )
        R = ml.reflectance()[0]
        T = ml.transmittance()[0]
        R_plus_T = R + T
        max_dev = abs(R_plus_T - 1.0)
        passed = abs(R_plus_T - 1.0) < tolerance
        results = (R, T, R_plus_T, max_dev, passed)
        
        if not passed:
            all_passed = False
            failing_cases.append((n_layers, angle, R, T, R_plus_T, max_dev))
        
        status = "✓ PASS" if passed else f"✗ FAIL"
        print(f"{angle:>6.0f}°   {R:<15.10f} {T:<15.10f} {R_plus_T:<15.10f} {status:<15} {max_dev:<15.2e}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

total_tests = (max_layers + 1) * len(angles)
passed_tests = total_tests - len(failing_cases)

print(f"\nTotal test cases: {total_tests}")
print(f"Passed: {passed_tests}")
print(f"Failed: {len(failing_cases)}")

if all_passed:
    print(f"\n✓ ALL TESTS PASSED!")
    print(f"  All cases satisfy energy conservation: R + T = 1.0")
    print(f"  Tested: 0 to {max_layers} layers, {len(angles)} oblique angles")
    print(f"  Tolerance: {tolerance}")
else:
    print(f"\n✗ SOME TESTS FAILED")
    print(f"  Total failing cases: {len(failing_cases)}")
    print("\n  Failing cases (first 20):")
    print(f"  {'Layers':<8} {'Angle':<8} {'R':<15} {'T':<15} {'R+T':<15} {'Deviation':<15}")
    print("  " + "-" * 80)
    for i, (n_layers, angle, R, T, r_plus_t, dev) in enumerate(failing_cases[:20]):
        print(f"  {n_layers:<8} {angle:>6.0f}°   {R:<15.10f} {T:<15.10f} {r_plus_t:<15.10f} {dev:<15.2e}")
    if len(failing_cases) > 20:
        print(f"  ... and {len(failing_cases) - 20} more cases")

print("\n" + "=" * 80)

# Additional analysis: Check if there's a pattern
if failing_cases:
    print("\nPattern Analysis:")
    print("=" * 80)
    
    # Group by number of layers
    by_layers = {}
    for n_layers, angle, R, T, r_plus_t, dev in failing_cases:
        if n_layers not in by_layers:
            by_layers[n_layers] = []
        by_layers[n_layers].append((angle, R, T, r_plus_t, dev))
    
    print("\nFailing cases grouped by number of layers:")
    for n_layers in sorted(by_layers.keys()):
        cases = by_layers[n_layers]
        avg_r_plus_t = np.mean([r_plus_t for _, _, _, r_plus_t, _ in cases])
        min_r_plus_t = np.min([r_plus_t for _, _, _, r_plus_t, _ in cases])
        max_r_plus_t = np.max([r_plus_t for _, _, _, r_plus_t, _ in cases])
        print(f"\n  {n_layers} layer(s): {len(cases)} failing cases")
        print(f"    R+T range: [{min_r_plus_t:.10f}, {max_r_plus_t:.10f}]")
        print(f"    Average R+T = {avg_r_plus_t:.10f}")
        print(f"    Angles: {', '.join([f'{a:.0f}°' for a, _, _, _, _ in cases])}")
    
    # Group by angle to see if values are consistent across layer counts
    print("\nR+T values by angle (checking consistency across layer counts):")
    print("-" * 80)
    print(f"{'Angle':<8} {'R+T (1 layer)':<18} {'R+T (2 layers)':<18} {'R+T (3 layers)':<18} {'Consistent?':<12}")
    print("-" * 80)
    
    by_angle = {}
    for n_layers, angle, R, T, r_plus_t, dev in failing_cases:
        if angle not in by_angle:
            by_angle[angle] = {}
        by_angle[angle][n_layers] = r_plus_t
    
    for angle in sorted(by_angle.keys()):
        values = by_angle[angle]
        r_plus_t_1 = values.get(1, None)
        r_plus_t_2 = values.get(2, None)
        r_plus_t_3 = values.get(3, None)
        
        if r_plus_t_1 is not None and r_plus_t_2 is not None and r_plus_t_3 is not None:
            consistent = abs(r_plus_t_1 - r_plus_t_2) < 1e-10 and abs(r_plus_t_2 - r_plus_t_3) < 1e-10
            consistent_str = "✓ Yes" if consistent else "✗ No"
            print(f"{angle:>6.0f}°   {r_plus_t_1:<18.10f} {r_plus_t_2:<18.10f} {r_plus_t_3:<18.10f} {consistent_str:<12}")

print("\n" + "=" * 80)

