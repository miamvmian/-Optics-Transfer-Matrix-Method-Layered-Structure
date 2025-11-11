#!/usr/bin/env python
"""
Comprehensive Energy Conservation Test

Tests R + T = 1.0 for various scenarios to verify the fixes.
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure

print("=" * 80)
print("Comprehensive Energy Conservation Test")
print("=" * 80)

tolerance = 1e-10
test_results = []

# Test 1: Normal incidence, single interface (0 layers)
print("\n" + "-" * 80)
print("Test 1: Normal Incidence, Single Interface (0 layers)")
print("-" * 80)

wavelengths = np.array([500e-9, 600e-9, 700e-9])
for pol in ['s', 'p']:
    ml = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[],
        eps_incident=1.0,
        eps_exit=2.25
    )
    R = ml.reflectance()
    T = ml.transmittance()
    R_plus_T = R + T
    max_dev = np.max(np.abs(R_plus_T - 1.0))
    passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
    test_results.append(("Normal, 0 layers", pol, passed, max_dev))
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {pol}-polarization: {status} | R+T = {R_plus_T} | Max deviation: {max_dev:.2e}")

# Test 2: Normal incidence, 1-10 layers
print("\n" + "-" * 80)
print("Test 2: Normal Incidence, 1-10 Layers")
print("-" * 80)

wavelengths = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
for n_layers in range(1, 11):
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 5e-9)
        if i % 2 == 0:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 2.13}))
        else:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 5.76}))
    
    for pol in ['s', 'p']:
        ml = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=0.0,
            polarization=pol,
            layers=layers,
            eps_incident=1.0,
            eps_exit=1.0
        )
        R = ml.reflectance()
        T = ml.transmittance()
        R_plus_T = R + T
        max_dev = np.max(np.abs(R_plus_T - 1.0))
        passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
        test_results.append((f"Normal, {n_layers} layers", pol, passed, max_dev))
        
        if not passed:
            status = "✗ FAIL"
            print(f"  {n_layers} layers, {pol}-pol: {status} | Max deviation: {max_dev:.2e} | R+T = {np.mean(R_plus_T):.10f}")

# Test 3: Oblique incidence, single interface
print("\n" + "-" * 80)
print("Test 3: Oblique Incidence (30°), Single Interface")
print("-" * 80)

wavelengths = np.array([500e-9, 600e-9, 700e-9])
for pol in ['s', 'p']:
    ml = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=30.0,
        polarization=pol,
        layers=[],
        eps_incident=1.0,
        eps_exit=2.25
    )
    R = ml.reflectance()
    T = ml.transmittance()
    R_plus_T = R + T
    max_dev = np.max(np.abs(R_plus_T - 1.0))
    passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
    test_results.append(("Oblique 30°, 0 layers", pol, passed, max_dev))
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {pol}-polarization: {status} | R+T = {R_plus_T} | Max deviation: {max_dev:.2e}")

# Test 4: Oblique incidence, 1-5 layers
print("\n" + "-" * 80)
print("Test 4: Oblique Incidence (30°), 1-5 Layers")
print("-" * 80)

wavelengths = np.array([500e-9, 600e-9, 700e-9])
for n_layers in range(1, 6):
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 10e-9)
        if i % 2 == 0:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 2.13}))
        else:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 5.76}))
    
    for pol in ['s', 'p']:
        ml = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=30.0,
            polarization=pol,
            layers=layers,
            eps_incident=1.0,
            eps_exit=1.0
        )
        R = ml.reflectance()
        T = ml.transmittance()
        R_plus_T = R + T
        max_dev = np.max(np.abs(R_plus_T - 1.0))
        passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
        test_results.append((f"Oblique 30°, {n_layers} layers", pol, passed, max_dev))
        
        if not passed:
            status = "✗ FAIL"
            print(f"  {n_layers} layers, {pol}-pol: {status} | Max deviation: {max_dev:.2e} | R+T = {np.mean(R_plus_T):.10f}")

# Test 5: Oblique incidence (45°), 1-3 layers
print("\n" + "-" * 80)
print("Test 5: Oblique Incidence (45°), 1-3 Layers")
print("-" * 80)

wavelengths = np.array([500e-9, 600e-9, 700e-9])
for n_layers in range(1, 4):
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 10e-9)
        if i % 2 == 0:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 2.13}))
        else:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 5.76}))
    
    for pol in ['s', 'p']:
        ml = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=45.0,
            polarization=pol,
            layers=layers,
            eps_incident=1.0,
            eps_exit=1.0
        )
        R = ml.reflectance()
        T = ml.transmittance()
        R_plus_T = R + T
        max_dev = np.max(np.abs(R_plus_T - 1.0))
        passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
        test_results.append((f"Oblique 45°, {n_layers} layers", pol, passed, max_dev))
        
        if not passed:
            status = "✗ FAIL"
            print(f"  {n_layers} layers, {pol}-pol: {status} | Max deviation: {max_dev:.2e} | R+T = {np.mean(R_plus_T):.10f}")

# Test 6: Different exit medium (this is the failing case - let's investigate)
print("\n" + "-" * 80)
print("Test 6: Normal Incidence, Different Exit Medium (Glass)")
print("-" * 80)

wavelengths = np.array([500e-9, 600e-9, 700e-9])
layers = [
    Layer(thickness=100e-9, optical_property={'type': 'permittivity', 'value': 2.13}),
    Layer(thickness=100e-9, optical_property={'type': 'permittivity', 'value': 5.76})
]

for pol in ['s', 'p']:
    ml = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=layers,
        eps_incident=1.0,
        eps_exit=2.25  # Glass substrate
    )
    R = ml.reflectance()
    T = ml.transmittance()
    R_plus_T = R + T
    max_dev = np.max(np.abs(R_plus_T - 1.0))
    passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
    test_results.append(("Normal, 2 layers, glass exit", pol, passed, max_dev))
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {pol}-polarization: {status} | R+T = {R_plus_T} | Max deviation: {max_dev:.2e}")
    if not passed:
        print(f"    R = {R}")
        print(f"    T = {T}")
        print(f"    Note: For different exit medium, R+T may not equal 1.0 due to impedance mismatch")
        print(f"    The correct formula accounts for power flow: T_corrected = (n_exit*cos(theta_exit))/(n_incident*cos(theta_incident)) * |t|²")

# Test 7: Oblique incidence with different exit medium
print("\n" + "-" * 80)
print("Test 7: Oblique Incidence (30°), Single Layer, Different Exit Medium")
print("-" * 80)

wavelengths = np.array([500e-9, 600e-9, 700e-9])
layer = Layer(thickness=100e-9, optical_property={'type': 'permittivity', 'value': 2.13})

for pol in ['s', 'p']:
    ml = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=30.0,
        polarization=pol,
        layers=[layer],
        eps_incident=1.0,
        eps_exit=2.25
    )
    R = ml.reflectance()
    T = ml.transmittance()
    R_plus_T = R + T
    max_dev = np.max(np.abs(R_plus_T - 1.0))
    passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
    test_results.append(("Oblique 30°, 1 layer, glass exit", pol, passed, max_dev))
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {pol}-polarization: {status} | R+T = {R_plus_T} | Max deviation: {max_dev:.2e}")

# Summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)

total_tests = len(test_results)
passed_tests = sum(1 for _, _, passed, _ in test_results if passed)
failed_tests = total_tests - passed_tests

print(f"\nTotal tests: {total_tests}")
print(f"Passed: {passed_tests}")
print(f"Failed: {failed_tests}")

if failed_tests > 0:
    print("\nFailed tests:")
    for test_name, pol, passed, max_dev in test_results:
        if not passed:
            print(f"  {test_name}, {pol}-polarization: Max deviation = {max_dev:.2e}")

# Group by scenario
scenarios = {}
for test_name, pol, passed, max_dev in test_results:
    if test_name not in scenarios:
        scenarios[test_name] = {"s": None, "p": None}
    scenarios[test_name][pol] = (passed, max_dev)

print("\n" + "-" * 80)
print("Results by Scenario")
print("-" * 80)
print(f"{'Scenario':<35} {'s-pol':<15} {'p-pol':<15}")
print("-" * 80)

for scenario, results in sorted(scenarios.items()):
    s_status = f"✓ ({results['s'][1]:.2e})" if results['s'] and results['s'][0] else f"✗ ({results['s'][1]:.2e})" if results['s'] else "N/A"
    p_status = f"✓ ({results['p'][1]:.2e})" if results['p'] and results['p'][0] else f"✗ ({results['p'][1]:.2e})" if results['p'] else "N/A"
    print(f"{scenario:<35} {s_status:<15} {p_status:<15}")

print("\n" + "=" * 80)
if failed_tests == 0:
    print("✓ ALL TESTS PASSED - Energy conservation verified!")
    print(f"  All {total_tests} test cases show R + T = 1.0 within tolerance ({tolerance})")
else:
    print(f"✗ SOME TESTS FAILED - {failed_tests} out of {total_tests} tests failed")
    print("\nNote: Failures with different exit media may be expected if the transmittance")
    print("calculation doesn't properly account for power flow correction at the exit interface.")
print("=" * 80)

exit(0 if failed_tests == 0 else 1)

