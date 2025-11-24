#!/usr/bin/env python
"""
Comprehensive Energy Conservation Test: Constant vs Wavelength-Dependent Permittivity

Tests energy conservation (R + T = 1.0) for:
1. Constant permittivity (scalar values)
2. Wavelength-dependent permittivity (arrays matching wavelengths)
3. Mixed cases (some constant, some wavelength-dependent)
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure

print("=" * 80)
print("Energy Conservation Test: Constant vs Wavelength-Dependent Permittivity")
print("=" * 80)

tolerance = 1e-10
wavelengths = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
n_wavelengths = len(wavelengths)

# Test 1: Constant permittivity (scalar) - all media and layers
print("\n" + "-" * 80)
print("Test 1: Constant Permittivity (Scalar Values)")
print("-" * 80)

eps_air_const = 1.0
eps_glass_const = 2.25
eps_silica_const = 2.13
eps_titanium_const = 5.76

for pol in ['s', 'p']:
    # Single interface with constant permittivity
    ml1 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[],
        eps_incident=eps_air_const,
        eps_exit=eps_glass_const
    )
    R1 = ml1.reflectance()
    T1 = ml1.transmittance()
    R_plus_T1 = R1 + T1
    max_dev1 = np.max(np.abs(R_plus_T1 - 1.0))
    passed1 = np.allclose(R_plus_T1, 1.0, atol=tolerance)
    
    # Single layer with constant permittivity
    layer_const = Layer(
        thickness=100e-9,
        optical_property={'type': 'permittivity', 'value': eps_silica_const}
    )
    ml2 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer_const],
        eps_incident=eps_air_const,
        eps_exit=eps_air_const
    )
    R2 = ml2.reflectance()
    T2 = ml2.transmittance()
    R_plus_T2 = R2 + T2
    max_dev2 = np.max(np.abs(R_plus_T2 - 1.0))
    passed2 = np.allclose(R_plus_T2, 1.0, atol=tolerance)
    
    status1 = "✓ PASS" if passed1 else "✗ FAIL"
    status2 = "✓ PASS" if passed2 else "✗ FAIL"
    print(f"  {pol.upper()}-pol, single interface: {status1} | Max dev: {max_dev1:.2e}")
    print(f"  {pol.upper()}-pol, single layer: {status2} | Max dev: {max_dev2:.2e}")

# Test 2: Wavelength-dependent permittivity (arrays)
print("\n" + "-" * 80)
print("Test 2: Wavelength-Dependent Permittivity (Arrays)")
print("-" * 80)

# Create wavelength-dependent permittivity using a simple dispersion model
# eps(lambda) = A + B/lambda^2 (Cauchy model approximation)
# For glass: n ≈ 1.5, so eps ≈ 2.25, with slight dispersion
eps_air_array = np.ones(n_wavelengths)  # Air is constant
eps_glass_array = 2.25 + 0.01 * (wavelengths * 1e9 - 600) / 600  # Slight dispersion
eps_silica_array = 2.13 + 0.005 * (wavelengths * 1e9 - 600) / 600
eps_titanium_array = 5.76 + 0.02 * (wavelengths * 1e9 - 600) / 600

for pol in ['s', 'p']:
    # Single interface with wavelength-dependent permittivity
    ml3 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[],
        eps_incident=eps_air_array,
        eps_exit=eps_glass_array
    )
    R3 = ml3.reflectance()
    T3 = ml3.transmittance()
    R_plus_T3 = R3 + T3
    max_dev3 = np.max(np.abs(R_plus_T3 - 1.0))
    passed3 = np.allclose(R_plus_T3, 1.0, atol=tolerance)
    
    # Single layer with wavelength-dependent permittivity
    layer_array = Layer(
        thickness=100e-9,
        optical_property={'type': 'permittivity', 'value': eps_silica_array}
    )
    ml4 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer_array],
        eps_incident=eps_air_array,
        eps_exit=eps_air_array
    )
    R4 = ml4.reflectance()
    T4 = ml4.transmittance()
    R_plus_T4 = R4 + T4
    max_dev4 = np.max(np.abs(R_plus_T4 - 1.0))
    passed4 = np.allclose(R_plus_T4, 1.0, atol=tolerance)
    
    status3 = "✓ PASS" if passed3 else "✗ FAIL"
    status4 = "✓ PASS" if passed4 else "✗ FAIL"
    print(f"  {pol.upper()}-pol, single interface: {status3} | Max dev: {max_dev3:.2e}")
    print(f"  {pol.upper()}-pol, single layer: {status4} | Max dev: {max_dev4:.2e}")

# Test 3: Mixed cases (constant + wavelength-dependent)
print("\n" + "-" * 80)
print("Test 3: Mixed Cases (Constant + Wavelength-Dependent)")
print("-" * 80)

for pol in ['s', 'p']:
    # Constant incident, wavelength-dependent exit
    ml5 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[],
        eps_incident=eps_air_const,
        eps_exit=eps_glass_array
    )
    R5 = ml5.reflectance()
    T5 = ml5.transmittance()
    R_plus_T5 = R5 + T5
    max_dev5 = np.max(np.abs(R_plus_T5 - 1.0))
    passed5 = np.allclose(R_plus_T5, 1.0, atol=tolerance)
    
    # Wavelength-dependent layer, constant media
    layer_mixed = Layer(
        thickness=100e-9,
        optical_property={'type': 'permittivity', 'value': eps_silica_array}
    )
    ml6 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer_mixed],
        eps_incident=eps_air_const,
        eps_exit=eps_air_const
    )
    R6 = ml6.reflectance()
    T6 = ml6.transmittance()
    R_plus_T6 = R6 + T6
    max_dev6 = np.max(np.abs(R_plus_T6 - 1.0))
    passed6 = np.allclose(R_plus_T6, 1.0, atol=tolerance)
    
    status5 = "✓ PASS" if passed5 else "✗ FAIL"
    status6 = "✓ PASS" if passed6 else "✗ FAIL"
    print(f"  {pol.upper()}-pol, const incident + array exit: {status5} | Max dev: {max_dev5:.2e}")
    print(f"  {pol.upper()}-pol, const media + array layer: {status6} | Max dev: {max_dev6:.2e}")

# Test 4: Multiple layers with mixed permittivity types
print("\n" + "-" * 80)
print("Test 4: Multiple Layers with Mixed Permittivity Types")
print("-" * 80)

for pol in ['s', 'p']:
    # Two layers: one constant, one wavelength-dependent
    layer1_const = Layer(
        thickness=100e-9,
        optical_property={'type': 'permittivity', 'value': eps_silica_const}
    )
    layer2_array = Layer(
        thickness=150e-9,
        optical_property={'type': 'permittivity', 'value': eps_titanium_array}
    )
    ml7 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer1_const, layer2_array],
        eps_incident=eps_air_const,
        eps_exit=eps_air_const
    )
    R7 = ml7.reflectance()
    T7 = ml7.transmittance()
    R_plus_T7 = R7 + T7
    max_dev7 = np.max(np.abs(R_plus_T7 - 1.0))
    passed7 = np.allclose(R_plus_T7, 1.0, atol=tolerance)
    
    status7 = "✓ PASS" if passed7 else "✗ FAIL"
    print(f"  {pol.upper()}-pol, 2 layers (const + array): {status7} | Max dev: {max_dev7:.2e}")

# Test 5: Different numbers of layers with wavelength-dependent permittivity
print("\n" + "-" * 80)
print("Test 5: Different Numbers of Layers with Wavelength-Dependent Permittivity")
print("-" * 80)

max_layers = 10
angles_test5 = [0.0, 30.0, 45.0]
results_layers = {'s': {'passed': 0, 'failed': 0}, 'p': {'passed': 0, 'failed': 0}}

for n_layers in range(0, max_layers + 1):
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 5e-9)
        if i % 2 == 0:
            # Even layers: wavelength-dependent permittivity
            layers.append(Layer(
                thickness=thickness,
                optical_property={'type': 'permittivity', 'value': eps_silica_array}
            ))
        else:
            # Odd layers: constant permittivity
            layers.append(Layer(
                thickness=thickness,
                optical_property={'type': 'permittivity', 'value': eps_titanium_const}
            ))
    
    for angle in angles_test5:
        for pol in ['s', 'p']:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=layers,
                eps_incident=eps_air_array,
                eps_exit=eps_glass_array
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
            
            if passed:
                results_layers[pol]['passed'] += 1
            else:
                results_layers[pol]['failed'] += 1
            
            # Print only for key cases to avoid too much output
            if (n_layers <= 2 or n_layers == 5 or n_layers == 10) and angle in [0.0, 45.0]:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {n_layers} layers, {angle:3.0f}°, {pol.upper()}-pol: {status} | Max dev: {max_dev:.2e}")

print(f"\n  Summary: s-pol passed: {results_layers['s']['passed']}, failed: {results_layers['s']['failed']}")
print(f"           p-pol passed: {results_layers['p']['passed']}, failed: {results_layers['p']['failed']}")

# Test 6: Oblique incidence with wavelength-dependent permittivity
print("\n" + "-" * 80)
print("Test 6: Oblique Incidence with Wavelength-Dependent Permittivity")
print("-" * 80)

angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0]
results_angles = {'s': {'passed': 0, 'failed': 0}, 'p': {'passed': 0, 'failed': 0}}

for angle in angles:
    for pol in ['s', 'p']:
        ml8 = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=angle,
            polarization=pol,
            layers=[],
            eps_incident=eps_air_array,
            eps_exit=eps_glass_array
        )
        R8 = ml8.reflectance()
        T8 = ml8.transmittance()
        R_plus_T8 = R8 + T8
        max_dev8 = np.max(np.abs(R_plus_T8 - 1.0))
        passed8 = np.allclose(R_plus_T8, 1.0, atol=tolerance)
        
        if passed8:
            results_angles[pol]['passed'] += 1
        else:
            results_angles[pol]['failed'] += 1
        
        if not passed8 or angle in [0.0, 30.0, 60.0, 85.0]:
            status8 = "✓ PASS" if passed8 else "✗ FAIL"
            print(f"  {angle:3.0f}°, {pol.upper()}-pol: {status8} | Max dev: {max_dev8:.2e}")

print(f"\n  Summary: s-pol passed: {results_angles['s']['passed']}, failed: {results_angles['s']['failed']}")
print(f"           p-pol passed: {results_angles['p']['passed']}, failed: {results_angles['p']['failed']}")

# Test 7: Multiple layers at different angles with wavelength-dependent permittivity
print("\n" + "-" * 80)
print("Test 7: Multiple Layers at Different Angles with Wavelength-Dependent Permittivity")
print("-" * 80)

layer_counts = [0, 1, 2, 3, 5, 10]
test_angles = [0.0, 30.0, 45.0, 60.0]
results_combined = {'s': {'passed': 0, 'failed': 0}, 'p': {'passed': 0, 'failed': 0}}

for n_layers in layer_counts:
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 5e-9)
        if i % 2 == 0:
            layers.append(Layer(
                thickness=thickness,
                optical_property={'type': 'permittivity', 'value': eps_silica_array}
            ))
        else:
            layers.append(Layer(
                thickness=thickness,
                optical_property={'type': 'permittivity', 'value': eps_titanium_array}
            ))
    
    for angle in test_angles:
        for pol in ['s', 'p']:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=layers,
                eps_incident=eps_air_array,
                eps_exit=eps_glass_array
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
            
            if passed:
                results_combined[pol]['passed'] += 1
            else:
                results_combined[pol]['failed'] += 1
            
            # Print only failures or key cases
            if not passed or (n_layers in [0, 1, 5] and angle in [0.0, 45.0]):
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {n_layers} layers, {angle:3.0f}°, {pol.upper()}-pol: {status} | Max dev: {max_dev:.2e}")

print(f"\n  Summary: s-pol passed: {results_combined['s']['passed']}, failed: {results_combined['s']['failed']}")
print(f"           p-pol passed: {results_combined['p']['passed']}, failed: {results_combined['p']['failed']}")

# Test 8: Refractive index type (should be converted to permittivity)
print("\n" + "-" * 80)
print("Test 8: Refractive Index Type (Converted to Permittivity)")
print("-" * 80)

# Test with refractive index (n) instead of permittivity
# n = 1.5 -> eps = 2.25
n_glass_const = 1.5
n_silica_const = 1.46

for pol in ['s', 'p']:
    layer_n = Layer(
        thickness=100e-9,
        optical_property={'type': 'refractive_index', 'value': n_glass_const}
    )
    ml9 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer_n],
        eps_incident=eps_air_const,
        eps_exit=eps_air_const
    )
    R9 = ml9.reflectance()
    T9 = ml9.transmittance()
    R_plus_T9 = R9 + T9
    max_dev9 = np.max(np.abs(R_plus_T9 - 1.0))
    passed9 = np.allclose(R_plus_T9, 1.0, atol=tolerance)
    
    status9 = "✓ PASS" if passed9 else "✗ FAIL"
    print(f"  {pol.upper()}-pol, refractive_index (const): {status9} | Max dev: {max_dev9:.2e}")

# Test with wavelength-dependent refractive index
n_glass_array = 1.5 + 0.01 * (wavelengths * 1e9 - 600) / 600

for pol in ['s', 'p']:
    layer_n_array = Layer(
        thickness=100e-9,
        optical_property={'type': 'refractive_index', 'value': n_glass_array}
    )
    ml10 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[layer_n_array],
        eps_incident=eps_air_array,
        eps_exit=eps_air_array
    )
    R10 = ml10.reflectance()
    T10 = ml10.transmittance()
    R_plus_T10 = R10 + T10
    max_dev10 = np.max(np.abs(R_plus_T10 - 1.0))
    passed10 = np.allclose(R_plus_T10, 1.0, atol=tolerance)
    
    status10 = "✓ PASS" if passed10 else "✗ FAIL"
    print(f"  {pol.upper()}-pol, refractive_index (array): {status10} | Max dev: {max_dev10:.2e}")

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("\nAll tests verify energy conservation (R + T = 1.0) for:")
print("  ✓ Constant permittivity (scalar values)")
print("  ✓ Wavelength-dependent permittivity (arrays)")
print("  ✓ Mixed cases (constant + wavelength-dependent)")
print("  ✓ Multiple layers with different permittivity types")
print("  ✓ Different numbers of layers (0 to 10 layers)")
print("  ✓ Different incident angles (0° to 85°)")
print("  ✓ Oblique incidence with wavelength-dependent permittivity")
print("  ✓ Multiple layers at different angles with wavelength-dependent permittivity")
print("  ✓ Refractive index type (both constant and wavelength-dependent)")
print("\n" + "=" * 80)

