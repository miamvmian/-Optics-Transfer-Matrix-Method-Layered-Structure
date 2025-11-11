#!/usr/bin/env python
"""
Comprehensive Energy Conservation Test: Both S and P Polarizations
Tests with different exit medium and oblique incidence
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure

print("=" * 80)
print("Energy Conservation Test: Both S and P Polarizations")
print("Different Incident/Exit Media, Oblique Incidence")
print("=" * 80)

tolerance = 1e-10
wavelengths = np.array([600e-9])
eps_air = 1.0
eps_glass = 2.25

# Test angles (oblique incidence)
angles = [15.0, 30.0, 45.0, 60.0, 75.0, 85.0]

# Test different numbers of layers
max_layers = 5

print(f"\nTesting from 0 to {max_layers} layers")
print(f"Angles: {', '.join([f'{a:.0f}°' for a in angles])}")
print(f"Incident medium: Air (eps = {eps_air})")
print(f"Exit medium: Glass (eps = {eps_glass})")
print("\n" + "=" * 80)

results_summary = {
    's': {'passed': 0, 'failed': 0, 'cases': []},
    'p': {'passed': 0, 'failed': 0, 'cases': []}
}

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
    print(f"{'Angle':<8} {'s-pol R+T':<18} {'s-pol Status':<15} {'p-pol R+T':<18} {'p-pol Status':<15}")
    print("-" * 80)
    
    for angle in angles:
        row_results = {}
        for pol in ['s', 'p']:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=layers,
                eps_incident=eps_air,
                eps_exit=eps_glass
            )
            R = ml.reflectance()[0]
            T = ml.transmittance()[0]
            R_plus_T = R + T
            max_dev = abs(R_plus_T - 1.0)
            passed = abs(R_plus_T - 1.0) < tolerance
            
            row_results[pol] = (R, T, R_plus_T, max_dev, passed)
            
            if passed:
                results_summary[pol]['passed'] += 1
            else:
                results_summary[pol]['failed'] += 1
                results_summary[pol]['cases'].append((n_layers, angle, R, T, R_plus_T, max_dev))
        
        s_r, s_t, s_r_plus_t, s_dev, s_passed = row_results['s']
        p_r, p_t, p_r_plus_t, p_dev, p_passed = row_results['p']
        
        s_status = "✓ PASS" if s_passed else f"✗ FAIL ({s_dev:.2e})"
        p_status = "✓ PASS" if p_passed else f"✗ FAIL ({p_dev:.2e})"
        
        print(f"{angle:>6.0f}°   {s_r_plus_t:<18.10f} {s_status:<15} {p_r_plus_t:<18.10f} {p_status:<15}")

print("\n" + "=" * 80)
print("Summary by Polarization")
print("=" * 80)

total_tests = (max_layers + 1) * len(angles)

for pol in ['s', 'p']:
    pol_name = "S-polarization (TE)" if pol == 's' else "P-polarization (TM)"
    passed = results_summary[pol]['passed']
    failed = results_summary[pol]['failed']
    
    print(f"\n{pol_name}:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {passed} ({100*passed/total_tests:.1f}%)")
    print(f"  Failed: {failed} ({100*failed/total_tests:.1f}%)")
    
    if failed > 0:
        print(f"\n  Failing cases (first 10):")
        print(f"  {'Layers':<8} {'Angle':<8} {'R':<15} {'T':<15} {'R+T':<15} {'Deviation':<15}")
        print("  " + "-" * 80)
        for i, (n_layers, angle, R, T, r_plus_t, dev) in enumerate(results_summary[pol]['cases'][:10]):
            print(f"  {n_layers:<8} {angle:>6.0f}°   {R:<15.10f} {T:<15.10f} {r_plus_t:<15.10f} {dev:<15.2e}")
        if len(results_summary[pol]['cases']) > 10:
            print(f"  ... and {len(results_summary[pol]['cases']) - 10} more cases")

print("\n" + "=" * 80)
print("Overall Summary")
print("=" * 80)

total_passed = results_summary['s']['passed'] + results_summary['p']['passed']
total_failed = results_summary['s']['failed'] + results_summary['p']['failed']
total_all = total_passed + total_failed

print(f"\nTotal test cases: {total_all}")
print(f"Total passed: {total_passed} ({100*total_passed/total_all:.1f}%)")
print(f"Total failed: {total_failed} ({100*total_failed/total_all:.1f}%)")

if total_failed == 0:
    print("\n✓ ALL TESTS PASSED - Energy conservation holds for all cases!")
else:
    print(f"\n✗ SOME TESTS FAILED - {total_failed} out of {total_all} tests failed")

print("\n" + "=" * 80)

# Quick check: Test same exit medium to verify it still works
print("\nVerification: Same Exit Medium (should all pass)")
print("=" * 80)

test_cases = [
    (0, 0.0, "Normal, single interface"),
    (0, 30.0, "Oblique 30°, single interface"),
    (1, 0.0, "Normal, 1 layer"),
    (1, 30.0, "Oblique 30°, 1 layer"),
    (2, 0.0, "Normal, 2 layers"),
    (2, 45.0, "Oblique 45°, 2 layers"),
]

for n_layers, angle, desc in test_cases:
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
        status = "✓ PASS" if dev < tolerance else "✗ FAIL"
        print(f"  {pol}-pol: R+T = {R_plus_T:.10f}, deviation = {dev:.2e} {status}")

print("\n" + "=" * 80)

