#!/usr/bin/env python
"""
Simple test for energy conservation: R + T = 1
Tests energy conservation for various configurations using the updated TMatrix.py API
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure

print("=" * 70)
print("Energy Conservation Test: R + T = 1 (for lossless systems)")
print("=" * 70)

tolerance = 1e-10

# Test 1: Single interface (zero layers)
print("\nTest 1: Single Interface (Zero Layers)")
wavelengths = np.array([500e-9])
structure1 = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=0.0,
    polarization="s",
    layers=[],
    eps_incident=1.0,
    eps_exit=1.0,
)
R1 = structure1.reflectance()[0]
T1 = structure1.transmittance()[0]
print(f"  eps_incident = eps_exit = 1.0")
print(f"  R = {R1:.10f}, T = {T1:.10f}, R + T = {R1 + T1:.10f}")
assert abs(R1 + T1 - 1.0) < tolerance, f"Energy not conserved! R + T = {R1 + T1}"
print("  ✓ PASSED")

# Test 2: Single interface with different media
print("\nTest 2: Single Interface, Different Media")
structure2 = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=0.0,
    polarization="s",
    layers=[],
    eps_incident=1.0,
    eps_exit=2.25,
)
R2 = structure2.reflectance()[0]
T2 = structure2.transmittance()[0]
print(f"  eps_incident = 1.0, eps_exit = 2.25")
print(f"  R = {R2:.10f}, T = {T2:.10f}, R + T = {R2 + T2:.10f}")
assert abs(R2 + T2 - 1.0) < tolerance, f"Energy not conserved! R + T = {R2 + T2}"
print("  ✓ PASSED")

# Test 3: Single layer
print("\nTest 3: Single Layer")
layers3 = [Layer(thickness=100e-9, optical_property={"type": "permittivity", "value": 2.25})]
structure3 = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=0.0,
    polarization="s",
    layers=layers3,
    eps_incident=1.0,
    eps_exit=1.0,
)
R3 = structure3.reflectance()[0]
T3 = structure3.transmittance()[0]
print(f"  Single layer: eps = 2.25, thickness = 100 nm")
print(f"  eps_incident = eps_exit = 1.0")
print(f"  R = {R3:.10f}, T = {T3:.10f}, R + T = {R3 + T3:.10f}")
if abs(R3 + T3 - 1.0) < tolerance:
    print("  ✓ PASSED")
else:
    print(f"  ✗ FAILED: Deviation = {abs(R3 + T3 - 1.0):.10f}")

# Test 4: Multiple layers
print("\nTest 4: Multiple Layers")
layers4 = [
    Layer(thickness=100e-9, optical_property={"type": "permittivity", "value": 2.25}),
    Layer(thickness=200e-9, optical_property={"type": "permittivity", "value": 2.25}),
]
structure4 = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=0.0,
    polarization="s",
    layers=layers4,
    eps_incident=1.0,
    eps_exit=1.0,
)
R4 = structure4.reflectance()[0]
T4 = structure4.transmittance()[0]
print(f"  2 layers: eps = [2.25, 2.25], thicknesses = [100, 200] nm")
print(f"  eps_incident = eps_exit = 1.0")
print(f"  R = {R4:.10f}, T = {T4:.10f}, R + T = {R4 + T4:.10f}")
if abs(R4 + T4 - 1.0) < tolerance:
    print("  ✓ PASSED")
else:
    print(f"  ✗ FAILED: Deviation = {abs(R4 + T4 - 1.0):.10f}")

# Test 5: Multiple wavelengths
print("\nTest 5: Multiple Wavelengths (Single Interface)")
wavelengths5 = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
structure5 = MultiLayerStructure(
    wavelengths=wavelengths5,
    angle_degrees=0.0,
    polarization="s",
    layers=[],
    eps_incident=1.0,
    eps_exit=1.0,
)
R5 = structure5.reflectance()
T5 = structure5.transmittance()
R_plus_T5 = R5 + T5
max_dev = np.max(np.abs(R_plus_T5 - 1.0))
print(f"  {len(wavelengths5)} wavelengths, single interface")
print(f"  R + T range: [{np.min(R_plus_T5):.10f}, {np.max(R_plus_T5):.10f}]")
print(f"  Max deviation from 1: {max_dev:.2e}")
if max_dev < tolerance:
    print("  ✓ PASSED")
else:
    print(f"  ✗ FAILED")

# Test 6: Both polarizations
print("\nTest 6: Both S and P Polarizations (Single Interface)")
for pol in ['s', 'p']:
    structure6 = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=0.0,
        polarization=pol,
        layers=[],
        eps_incident=1.0,
        eps_exit=2.25,
    )
    R6 = structure6.reflectance()[0]
    T6 = structure6.transmittance()[0]
    R_plus_T6 = R6 + T6
    dev = abs(R_plus_T6 - 1.0)
    status = "✓ PASSED" if dev < tolerance else "✗ FAILED"
    print(f"  {pol.upper()}-polarization: R+T = {R_plus_T6:.10f}, deviation = {dev:.2e} {status}")

# Test 7: Wavelength-dependent permittivity (constant scalar)
print("\nTest 7: Constant Permittivity (Scalar) - Multiple Wavelengths")
wavelengths7 = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
structure7 = MultiLayerStructure(
    wavelengths=wavelengths7,
    angle_degrees=0.0,
    polarization="s",
    layers=[],
    eps_incident=1.0,  # Constant scalar
    eps_exit=2.25,     # Constant scalar
)
R7 = structure7.reflectance()
T7 = structure7.transmittance()
R_plus_T7 = R7 + T7
max_dev7 = np.max(np.abs(R_plus_T7 - 1.0))
print(f"  Constant permittivity (scalar) broadcast to {len(wavelengths7)} wavelengths")
print(f"  R + T range: [{np.min(R_plus_T7):.10f}, {np.max(R_plus_T7):.10f}]")
print(f"  Max deviation from 1: {max_dev7:.2e}")
if max_dev7 < tolerance:
    print("  ✓ PASSED")
else:
    print(f"  ✗ FAILED")

# Test 8: Wavelength-dependent permittivity (arrays)
print("\nTest 8: Wavelength-Dependent Permittivity (Arrays)")
wavelengths8 = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
# Simple dispersion model: eps(lambda) = A + B/lambda^2
eps_air_array = np.ones(len(wavelengths8))  # Air is constant
eps_glass_array = 2.25 + 0.01 * (wavelengths8 * 1e9 - 600) / 600  # Slight dispersion

structure8 = MultiLayerStructure(
    wavelengths=wavelengths8,
    angle_degrees=0.0,
    polarization="s",
    layers=[],
    eps_incident=eps_air_array,   # Wavelength-dependent array
    eps_exit=eps_glass_array,     # Wavelength-dependent array
)
R8 = structure8.reflectance()
T8 = structure8.transmittance()
R_plus_T8 = R8 + T8
max_dev8 = np.max(np.abs(R_plus_T8 - 1.0))
print(f"  Wavelength-dependent permittivity arrays ({len(wavelengths8)} wavelengths)")
print(f"  eps_incident: constant (air)")
print(f"  eps_exit: varies from {eps_glass_array[0]:.4f} to {eps_glass_array[-1]:.4f}")
print(f"  R + T range: [{np.min(R_plus_T8):.10f}, {np.max(R_plus_T8):.10f}]")
print(f"  Max deviation from 1: {max_dev8:.2e}")
if max_dev8 < tolerance:
    print("  ✓ PASSED")
else:
    print(f"  ✗ FAILED")

# Test 9: Layer with wavelength-dependent permittivity
print("\nTest 9: Layer with Wavelength-Dependent Permittivity")
wavelengths9 = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
eps_silica_array = 2.13 + 0.005 * (wavelengths9 * 1e9 - 600) / 600

layer9 = Layer(
    thickness=100e-9,
    optical_property={'type': 'permittivity', 'value': eps_silica_array}  # Array
)
structure9 = MultiLayerStructure(
    wavelengths=wavelengths9,
    angle_degrees=0.0,
    polarization="s",
    layers=[layer9],
    eps_incident=1.0,  # Constant
    eps_exit=1.0,      # Constant
)
R9 = structure9.reflectance()
T9 = structure9.transmittance()
R_plus_T9 = R9 + T9
max_dev9 = np.max(np.abs(R_plus_T9 - 1.0))
print(f"  Layer permittivity varies from {eps_silica_array[0]:.4f} to {eps_silica_array[-1]:.4f}")
print(f"  R + T range: [{np.min(R_plus_T9):.10f}, {np.max(R_plus_T9):.10f}]")
print(f"  Max deviation from 1: {max_dev9:.2e}")
if max_dev9 < tolerance:
    print("  ✓ PASSED")
else:
    print(f"  ✗ FAILED")

# Test 10: Mixed constant and wavelength-dependent
print("\nTest 10: Mixed Constant and Wavelength-Dependent Permittivity")
wavelengths10 = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
eps_glass_array10 = 2.25 + 0.01 * (wavelengths10 * 1e9 - 600) / 600

structure10 = MultiLayerStructure(
    wavelengths=wavelengths10,
    angle_degrees=0.0,
    polarization="s",
    layers=[],
    eps_incident=1.0,           # Constant scalar
    eps_exit=eps_glass_array10,  # Wavelength-dependent array
)
R10 = structure10.reflectance()
T10 = structure10.transmittance()
R_plus_T10 = R10 + T10
max_dev10 = np.max(np.abs(R_plus_T10 - 1.0))
print(f"  Constant incident, wavelength-dependent exit")
print(f"  R + T range: [{np.min(R_plus_T10):.10f}, {np.max(R_plus_T10):.10f}]")
print(f"  Max deviation from 1: {max_dev10:.2e}")
if max_dev10 < tolerance:
    print("  ✓ PASSED")
else:
    print(f"  ✗ FAILED")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)

