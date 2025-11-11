#!/usr/bin/env python
"""
Focused Energy Conservation Test - Check specific cases
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure

print("=" * 80)
print("Focused Energy Conservation Test")
print("=" * 80)

tolerance = 1e-10
wavelengths = np.array([600e-9])

# Test cases to check
test_cases = [
    # (description, n_layers, angle, eps_incident, eps_exit, expected_to_pass)
    ("0 layers, normal, same exit", 0, 0.0, 1.0, 1.0, True),
    ("0 layers, normal, glass exit", 0, 0.0, 1.0, 2.25, True),  # Single interface should work
    ("0 layers, oblique 30°, same exit", 0, 30.0, 1.0, 1.0, True),
    ("0 layers, oblique 30°, glass exit", 0, 30.0, 1.0, 2.25, True),  # Single interface should work
    
    ("1 layer, normal, same exit", 1, 0.0, 1.0, 1.0, True),
    ("1 layer, normal, glass exit", 1, 0.0, 1.0, 2.25, False),  # Expected to fail
    ("1 layer, oblique 30°, same exit", 1, 30.0, 1.0, 1.0, True),
    ("1 layer, oblique 30°, glass exit", 1, 30.0, 1.0, 2.25, False),  # Expected to fail
    
    ("2 layers, normal, same exit", 2, 0.0, 1.0, 1.0, True),
    ("2 layers, normal, glass exit", 2, 0.0, 1.0, 2.25, False),  # Expected to fail
    ("2 layers, oblique 30°, same exit", 2, 30.0, 1.0, 1.0, True),
    ("2 layers, oblique 30°, glass exit", 2, 30.0, 1.0, 2.25, False),  # Expected to fail
    
    ("3 layers, normal, same exit", 3, 0.0, 1.0, 1.0, True),
    ("3 layers, normal, glass exit", 3, 0.0, 1.0, 2.25, False),  # Might fail
    ("3 layers, oblique 30°, same exit", 3, 30.0, 1.0, 1.0, True),
    ("3 layers, oblique 30°, glass exit", 3, 30.0, 1.0, 2.25, False),  # Might fail
]

print("\nTesting all cases...")
print("-" * 80)
print(f"{'Case':<40} {'s-pol R+T':<15} {'p-pol R+T':<15} {'Status':<10}")
print("-" * 80)

failing_cases = []

for desc, n_layers, angle, eps_incident, eps_exit, expected_to_pass in test_cases:
    # Create layers
    layers = []
    for i in range(n_layers):
        thickness = 100e-9
        if i % 2 == 0:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 2.13}))
        else:
            layers.append(Layer(thickness=thickness, optical_property={'type': 'permittivity', 'value': 5.76}))
    
    # Test both polarizations
    results = {}
    for pol in ['s', 'p']:
        ml = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=angle,
            polarization=pol,
            layers=layers,
            eps_incident=eps_incident,
            eps_exit=eps_exit
        )
        R = ml.reflectance()[0]
        T = ml.transmittance()[0]
        R_plus_T = R + T
        max_dev = abs(R_plus_T - 1.0)
        passed = abs(R_plus_T - 1.0) < tolerance
        results[pol] = (R_plus_T, max_dev, passed)
    
    s_r_plus_t, s_dev, s_passed = results['s']
    p_r_plus_t, p_dev, p_passed = results['p']
    
    both_passed = s_passed and p_passed
    status = "✓ PASS" if both_passed else "✗ FAIL"
    
    if not both_passed:
        failing_cases.append((desc, n_layers, angle, eps_incident, eps_exit, s_r_plus_t, p_r_plus_t))
    
    print(f"{desc:<40} {s_r_plus_t:<15.10f} {p_r_plus_t:<15.10f} {status:<10}")

print("\n" + "=" * 80)
print("Summary of Failing Cases")
print("=" * 80)

if failing_cases:
    print(f"\nTotal failing cases: {len(failing_cases)}")
    print("\nFailing cases breakdown:")
    print("-" * 80)
    print(f"{'Case':<40} {'n_layers':<10} {'angle':<8} {'exit medium':<15} {'s-pol R+T':<15} {'p-pol R+T':<15}")
    print("-" * 80)
    
    # Group by number of layers
    by_layers = {}
    for desc, n_layers, angle, eps_incident, eps_exit, s_r_plus_t, p_r_plus_t in failing_cases:
        if n_layers not in by_layers:
            by_layers[n_layers] = []
        exit_desc = "Glass" if eps_exit == 2.25 else "Same"
        by_layers[n_layers].append((desc, angle, exit_desc, s_r_plus_t, p_r_plus_t))
    
    for n_layers in sorted(by_layers.keys()):
        print(f"\n{n_layers} layer(s):")
        for desc, angle, exit_desc, s_r_plus_t, p_r_plus_t in by_layers[n_layers]:
            print(f"  {desc:<38} {angle:>6.0f}° {exit_desc:<15} {s_r_plus_t:<15.10f} {p_r_plus_t:<15.10f}")
    
    print("\n" + "=" * 80)
    print("CONFIRMATION:")
    print("=" * 80)
    
    # Check if only 1 and 2 layers fail
    failing_layer_counts = set(n_layers for _, n_layers, _, _, _, _, _ in failing_cases)
    print(f"\nFailing cases occur for: {sorted(failing_layer_counts)} layer(s)")
    
    if failing_layer_counts == {1, 2}:
        print("\n✓ CONFIRMED: Only 1-layer and 2-layer cases fail energy conservation")
        print("  (All failures are for cases with different exit medium - glass substrate)")
    else:
        print(f"\n✗ NOT CONFIRMED: Failures occur for {sorted(failing_layer_counts)} layer(s)")
        print("  (Expected only 1 and 2 layers)")
else:
    print("\n✓ ALL TESTS PASSED - No failing cases!")

print("=" * 80)

