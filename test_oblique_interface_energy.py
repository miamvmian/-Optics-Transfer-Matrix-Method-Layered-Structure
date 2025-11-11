#!/usr/bin/env python
"""
Energy Conservation Test for Oblique Incidence with Single Interface

Tests R + T = 1.0 for lossless structures at oblique incidence angles
with a single interface (no middle layers).
"""

import numpy as np
import matplotlib.pyplot as plt
from TMatrix import Layer, MultiLayerStructure

print("=" * 80)
print("Energy Conservation Test: Oblique Incidence, Single Interface (0 layers)")
print("=" * 80)

# Test parameters
tolerance = 1e-10
wavelengths = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])

# Material properties (lossless - real permittivities)
eps_air = 1.0
eps_glass = 2.25  # n = 1.5
eps_silicon = 12.25  # n = 3.5
eps_water = 1.77  # n ≈ 1.33

print(f"\nTest Configuration:")
print(f"  Wavelengths: {', '.join([f'{w*1e9:.0f}' for w in wavelengths])} nm")
print(f"  Tolerance: {tolerance}")
print(f"  Materials: Air (ε=1.0), Glass (ε=2.25), Silicon (ε=12.25), Water (ε=1.77)")

# Test different angles
angles_degrees = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 89])

# Store results
results = {
    'angle': [],
    'material': [],
    's_pol': {'R': [], 'T': [], 'R_plus_T': [], 'max_deviation': []},
    'p_pol': {'R': [], 'T': [], 'R_plus_T': [], 'max_deviation': []}
}

# Test 1: Air | Glass interface
print("\n" + "-" * 80)
print("Test 1: Air | Glass Interface")
print("-" * 80)

for angle in angles_degrees:
    for pol in ['s', 'p']:
        try:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=[],
                eps_incident=eps_air,
                eps_exit=eps_glass
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
            
            results['angle'].append(angle)
            results['material'].append('Air|Glass')
            results[f'{pol}_pol']['R'].append(np.mean(R))
            results[f'{pol}_pol']['T'].append(np.mean(T))
            results[f'{pol}_pol']['R_plus_T'].append(np.mean(R_plus_T))
            results[f'{pol}_pol']['max_deviation'].append(max_dev)
            
            if not passed or angle in [0, 30, 60, 85]:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {angle:3.0f}°, {pol.upper()}-pol: {status} | R+T = {np.mean(R_plus_T):.10f} | Max dev: {max_dev:.2e}")
        except Exception as e:
            print(f"  {angle:3.0f}°, {pol.upper()}-pol: ERROR - {str(e)}")

# Test 2: Air | Silicon interface
print("\n" + "-" * 80)
print("Test 2: Air | Silicon Interface")
print("-" * 80)

for angle in angles_degrees:
    for pol in ['s', 'p']:
        try:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=[],
                eps_incident=eps_air,
                eps_exit=eps_silicon
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
            
            results['angle'].append(angle)
            results['material'].append('Air|Silicon')
            results[f'{pol}_pol']['R'].append(np.mean(R))
            results[f'{pol}_pol']['T'].append(np.mean(T))
            results[f'{pol}_pol']['R_plus_T'].append(np.mean(R_plus_T))
            results[f'{pol}_pol']['max_deviation'].append(max_dev)
            
            if not passed or angle in [0, 30, 60, 85]:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {angle:3.0f}°, {pol.upper()}-pol: {status} | R+T = {np.mean(R_plus_T):.10f} | Max dev: {max_dev:.2e}")
        except Exception as e:
            print(f"  {angle:3.0f}°, {pol.upper()}-pol: ERROR - {str(e)}")

# Test 3: Air | Water interface
print("\n" + "-" * 80)
print("Test 3: Air | Water Interface")
print("-" * 80)

for angle in angles_degrees:
    for pol in ['s', 'p']:
        try:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=[],
                eps_incident=eps_air,
                eps_exit=eps_water
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
            
            results['angle'].append(angle)
            results['material'].append('Air|Water')
            results[f'{pol}_pol']['R'].append(np.mean(R))
            results[f'{pol}_pol']['T'].append(np.mean(T))
            results[f'{pol}_pol']['R_plus_T'].append(np.mean(R_plus_T))
            results[f'{pol}_pol']['max_deviation'].append(max_dev)
            
            if not passed or angle in [0, 30, 60, 85]:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {angle:3.0f}°, {pol.upper()}-pol: {status} | R+T = {np.mean(R_plus_T):.10f} | Max dev: {max_dev:.2e}")
        except Exception as e:
            print(f"  {angle:3.0f}°, {pol.upper()}-pol: ERROR - {str(e)}")

# Test 4: Glass | Air interface (reverse direction)
print("\n" + "-" * 80)
print("Test 4: Glass | Air Interface (Reverse Direction)")
print("-" * 80)

for angle in angles_degrees:
    for pol in ['s', 'p']:
        try:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=[],
                eps_incident=eps_glass,
                eps_exit=eps_air
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
            
            results['angle'].append(angle)
            results['material'].append('Glass|Air')
            results[f'{pol}_pol']['R'].append(np.mean(R))
            results[f'{pol}_pol']['T'].append(np.mean(T))
            results[f'{pol}_pol']['R_plus_T'].append(np.mean(R_plus_T))
            results[f'{pol}_pol']['max_deviation'].append(max_dev)
            
            if not passed or angle in [0, 30, 60, 85]:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {angle:3.0f}°, {pol.upper()}-pol: {status} | R+T = {np.mean(R_plus_T):.10f} | Max dev: {max_dev:.2e}")
        except Exception as e:
            print(f"  {angle:3.0f}°, {pol.upper()}-pol: ERROR - {str(e)}")

# Summary
print("\n" + "=" * 80)
print("Summary: Energy Conservation (R + T = 1.0) for Single Interface")
print("=" * 80)

# Group results by material
materials = ['Air|Glass', 'Air|Silicon', 'Air|Water', 'Glass|Air']
unique_angles = sorted(set(results['angle']))

print(f"\n{'Angle':<8} {'Material':<15} {'s-pol R+T':<15} {'s-pol Max Dev':<15} {'p-pol R+T':<15} {'p-pol Max Dev':<15} {'Status':<10}")
print("-" * 100)

all_passed = True
for material in materials:
    for angle in unique_angles:
        # Find matching results
        s_r_plus_t = None
        s_max_dev = None
        p_r_plus_t = None
        p_max_dev = None
        
        for i, (a, m) in enumerate(zip(results['angle'], results['material'])):
            if a == angle and m == material:
                if i < len(results['s_pol']['R_plus_T']):
                    s_r_plus_t = results['s_pol']['R_plus_T'][i]
                    s_max_dev = results['s_pol']['max_deviation'][i]
                if i < len(results['p_pol']['R_plus_T']):
                    p_r_plus_t = results['p_pol']['R_plus_T'][i]
                    p_max_dev = results['p_pol']['max_deviation'][i]
                break
        
        if s_r_plus_t is not None and p_r_plus_t is not None:
            s_passed = np.allclose(s_r_plus_t, 1.0, atol=tolerance)
            p_passed = np.allclose(p_r_plus_t, 1.0, atol=tolerance)
            status = "✓ PASS" if (s_passed and p_passed) else "✗ FAIL"
            if not (s_passed and p_passed):
                all_passed = False
            
            # Only print if failed or for key angles
            if not (s_passed and p_passed) or angle in [0, 30, 60, 85]:
                print(f"{angle:<8.0f} {material:<15} {s_r_plus_t:<15.10f} {s_max_dev:<15.2e} {p_r_plus_t:<15.10f} {p_max_dev:<15.2e} {status:<10}")

print("\n" + "=" * 80)
if all_passed:
    print("✓ ALL TESTS PASSED - Energy conservation verified for oblique incidence single interface!")
    print(f"  All test cases show R + T = 1.0 within tolerance ({tolerance})")
else:
    print("✗ SOME TESTS FAILED - Energy conservation not satisfied")
print("=" * 80)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: R + T vs Angle for Air|Glass
ax = axes[0, 0]
air_glass_angles = []
air_glass_s_r_plus_t = []
air_glass_p_r_plus_t = []
for i, (angle, material) in enumerate(zip(results['angle'], results['material'])):
    if material == 'Air|Glass':
        if angle not in air_glass_angles:
            air_glass_angles.append(angle)
            idx = len(air_glass_angles) - 1
            if i < len(results['s_pol']['R_plus_T']):
                air_glass_s_r_plus_t.append(results['s_pol']['R_plus_T'][i])
            if i < len(results['p_pol']['R_plus_T']):
                air_glass_p_r_plus_t.append(results['p_pol']['R_plus_T'][i])

air_glass_angles = sorted(air_glass_angles)
air_glass_s_r_plus_t = [results['s_pol']['R_plus_T'][i] for i, (a, m) in enumerate(zip(results['angle'], results['material'])) if m == 'Air|Glass' and a in air_glass_angles]
air_glass_p_r_plus_t = [results['p_pol']['R_plus_T'][i] for i, (a, m) in enumerate(zip(results['angle'], results['material'])) if m == 'Air|Glass' and a in air_glass_angles]

if air_glass_angles:
    ax.plot(air_glass_angles, air_glass_s_r_plus_t[:len(air_glass_angles)], 'o-', label='s-polarization (TE)', linewidth=2, markersize=6)
    ax.plot(air_glass_angles, air_glass_p_r_plus_t[:len(air_glass_angles)], 's-', label='p-polarization (TM)', linewidth=2, markersize=6)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='R + T = 1 (ideal)')
ax.set_xlabel('Incident Angle (degrees)', fontsize=12)
ax.set_ylabel('R + T', fontsize=12)
ax.set_title('Energy Conservation: Air | Glass Interface', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 92)

# Plot 2: Reflectance vs Angle for Air|Glass
ax = axes[0, 1]
air_glass_s_r = [results['s_pol']['R'][i] for i, (a, m) in enumerate(zip(results['angle'], results['material'])) if m == 'Air|Glass']
air_glass_p_r = [results['p_pol']['R'][i] for i, (a, m) in enumerate(zip(results['angle'], results['material'])) if m == 'Air|Glass']
if air_glass_angles and air_glass_s_r:
    ax.plot(air_glass_angles, air_glass_s_r[:len(air_glass_angles)], 'o-', label='s-polarization (TE)', linewidth=2, markersize=6)
    ax.plot(air_glass_angles, air_glass_p_r[:len(air_glass_angles)], 's-', label='p-polarization (TM)', linewidth=2, markersize=6)
ax.set_xlabel('Incident Angle (degrees)', fontsize=12)
ax.set_ylabel('Reflectance (R)', fontsize=12)
ax.set_title('Reflectance vs Angle: Air | Glass Interface', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 92)

# Plot 3: Maximum Deviation vs Angle
ax = axes[1, 0]
air_glass_s_max_dev = [results['s_pol']['max_deviation'][i] for i, (a, m) in enumerate(zip(results['angle'], results['material'])) if m == 'Air|Glass']
air_glass_p_max_dev = [results['p_pol']['max_deviation'][i] for i, (a, m) in enumerate(zip(results['angle'], results['material'])) if m == 'Air|Glass']
if air_glass_angles and air_glass_s_max_dev:
    ax.semilogy(air_glass_angles, air_glass_s_max_dev[:len(air_glass_angles)], 'o-', label='s-polarization (TE)', linewidth=2, markersize=6)
    ax.semilogy(air_glass_angles, air_glass_p_max_dev[:len(air_glass_angles)], 's-', label='p-polarization (TM)', linewidth=2, markersize=6)
ax.axhline(y=tolerance, color='red', linestyle='--', linewidth=2, label=f'Tolerance ({tolerance})')
ax.set_xlabel('Incident Angle (degrees)', fontsize=12)
ax.set_ylabel('Max Deviation from 1.0 (log scale)', fontsize=12)
ax.set_title('Maximum Deviation from Energy Conservation', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 92)

# Plot 4: Comparison of different materials at 30°
ax = axes[1, 1]
angle_30 = 30.0
materials_30 = []
s_r_plus_t_30 = []
p_r_plus_t_30 = []

for material in materials:
    for i, (angle, mat) in enumerate(zip(results['angle'], results['material'])):
        if angle == angle_30 and mat == material:
            if i < len(results['s_pol']['R_plus_T']):
                materials_30.append(material)
                s_r_plus_t_30.append(results['s_pol']['R_plus_T'][i])
                p_r_plus_t_30.append(results['p_pol']['R_plus_T'][i])
            break

if materials_30:
    x_pos = np.arange(len(materials_30))
    width = 0.35
    ax.bar(x_pos - width/2, s_r_plus_t_30, width, label='s-polarization (TE)', alpha=0.8)
    ax.bar(x_pos + width/2, p_r_plus_t_30, width, label='p-polarization (TM)', alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='R + T = 1 (ideal)')
    ax.set_xlabel('Material Interface', fontsize=12)
    ax.set_ylabel('R + T', fontsize=12)
    ax.set_title(f'Energy Conservation at {angle_30}° Incidence', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(materials_30, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('energy_conservation_oblique_interface.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: energy_conservation_oblique_interface.png")
plt.show()

print("\nTest completed!")

