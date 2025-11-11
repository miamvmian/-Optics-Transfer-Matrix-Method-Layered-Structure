#!/usr/bin/env python
"""
Energy Conservation Test: Zero Layers (Single Interface)
Oblique Incidence, Both S and P Polarizations
"""

import numpy as np
from TMatrix import MultiLayerStructure

print("=" * 80)
print("Energy Conservation Test: Zero Layers (Single Interface)")
print("Oblique Incidence, Both S and P Polarizations")
print("=" * 80)

tolerance = 1e-10
wavelengths = np.array([600e-9])
eps_air = 1.0
eps_glass = 2.25

# Test angles (oblique incidence)
angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0]

print(f"\nTesting single interface (0 layers)")
print(f"Angles: {', '.join([f'{a:.0f}°' for a in angles])}")
print(f"Incident medium: Air (eps = {eps_air})")
print(f"Exit medium: Glass (eps = {eps_glass})")
print("\n" + "=" * 80)

all_passed = True
failing_cases = []

for angle in angles:
    print(f"\nAngle: {angle:.0f}°")
    print("-" * 80)
    print(f"{'Polarization':<15} {'R':<15} {'T':<15} {'R+T':<15} {'Status':<15} {'Deviation':<15}")
    print("-" * 80)
    
    for pol in ['s', 'p']:
        ml = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=angle,
            polarization=pol,
            layers=[],
            eps_incident=eps_air,
            eps_exit=eps_glass
        )
        R = ml.reflectance()[0]
        T = ml.transmittance()[0]
        R_plus_T = R + T
        max_dev = abs(R_plus_T - 1.0)
        passed = abs(R_plus_T - 1.0) < tolerance
        
        if not passed:
            all_passed = False
            failing_cases.append((angle, pol, R, T, R_plus_T, max_dev))
        
        status = "✓ PASS" if passed else f"✗ FAIL"
        pol_name = "S (TE)" if pol == 's' else "P (TM)"
        print(f"{pol_name:<15} {R:<15.10f} {T:<15.10f} {R_plus_T:<15.10f} {status:<15} {max_dev:<15.2e}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

total_tests = len(angles) * 2  # 2 polarizations
passed_tests = total_tests - len(failing_cases)

print(f"\nTotal test cases: {total_tests}")
print(f"Passed: {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
print(f"Failed: {len(failing_cases)} ({100*len(failing_cases)/total_tests:.1f}%)")

if all_passed:
    print(f"\n✓ ALL TESTS PASSED!")
    print(f"  All cases satisfy energy conservation: R + T = 1.0")
    print(f"  Tested: {len(angles)} oblique angles, both s and p polarizations")
    print(f"  Tolerance: {tolerance}")
else:
    print(f"\n✗ SOME TESTS FAILED")
    print(f"  Total failing cases: {len(failing_cases)}")
    print("\n  Failing cases:")
    print(f"  {'Angle':<8} {'Pol':<6} {'R':<15} {'T':<15} {'R+T':<15} {'Deviation':<15}")
    print("  " + "-" * 80)
    for angle, pol, R, T, r_plus_t, dev in failing_cases:
        pol_name = "S" if pol == 's' else "P"
        print(f"  {angle:>6.0f}°   {pol_name:<6} {R:<15.10f} {T:<15.10f} {r_plus_t:<15.10f} {dev:<15.2e}")

print("\n" + "=" * 80)

# Additional verification: Test with same exit medium
print("\nVerification: Same Exit Medium (should all pass)")
print("=" * 80)

print(f"\n{'Angle':<8} {'s-pol R+T':<15} {'p-pol R+T':<15} {'Status':<10}")
print("-" * 60)

for angle in angles:
    results = {}
    for pol in ['s', 'p']:
        ml = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=angle,
            polarization=pol,
            layers=[],
            eps_incident=eps_air,
            eps_exit=eps_air
        )
        R = ml.reflectance()[0]
        T = ml.transmittance()[0]
        R_plus_T = R + T
        results[pol] = R_plus_T
    
    s_r_plus_t = results['s']
    p_r_plus_t = results['p']
    both_passed = abs(s_r_plus_t - 1.0) < tolerance and abs(p_r_plus_t - 1.0) < tolerance
    status = "✓ PASS" if both_passed else "✗ FAIL"
    print(f"{angle:>6.0f}°   {s_r_plus_t:<15.10f} {p_r_plus_t:<15.10f} {status:<10}")

print("\n" + "=" * 80)

