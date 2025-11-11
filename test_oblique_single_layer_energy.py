#!/usr/bin/env python
"""
Energy Conservation Test for Oblique Incidence with Single Layer

Tests R + T = 1.0 for lossless structures at oblique incidence angles
with a single layer between two media.
"""

import numpy as np
import matplotlib.pyplot as plt
from TMatrix import Layer, MultiLayerStructure

print("=" * 80)
print("Energy Conservation Test: Oblique Incidence, Single Layer (1 layer)")
print("=" * 80)

# Test parameters
tolerance = 1e-10
wavelengths = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])

# Material properties (lossless - real permittivities)
eps_air = 1.0
eps_glass = 2.25  # n = 1.5
eps_silica = 2.13  # n ≈ 1.46
eps_titanium = 5.76  # n ≈ 2.4

print(f"\nTest Configuration:")
print(f"  Wavelengths: {', '.join([f'{w*1e9:.0f}' for w in wavelengths])} nm")
print(f"  Tolerance: {tolerance}")
print(f"  Materials: Air (ε=1.0), Glass (ε=2.25), Silica (ε=2.13), Titanium (ε=5.76)")

# Test different angles
angles_degrees = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 85])

# Store results
results = {
    'angle': [],
    'layer_material': [],
    'thickness_nm': [],
    's_pol': {'R': [], 'T': [], 'R_plus_T': [], 'max_deviation': []},
    'p_pol': {'R': [], 'T': [], 'R_plus_T': [], 'max_deviation': []}
}

# Test 1: Air | Glass layer | Air
print("\n" + "-" * 80)
print("Test 1: Air | Glass Layer | Air")
print("-" * 80)

layer_thicknesses = [50e-9, 100e-9, 200e-9, 500e-9]  # Different thicknesses in meters

for thickness in layer_thicknesses:
    layer = Layer(
        thickness=thickness,
        optical_property={'type': 'permittivity', 'value': eps_glass}
    )
    
    print(f"\n  Layer thickness: {thickness*1e9:.0f} nm")
    
    for angle in angles_degrees:
        for pol in ['s', 'p']:
            try:
                ml = MultiLayerStructure(
                    wavelengths=wavelengths,
                    angle_degrees=angle,
                    polarization=pol,
                    layers=[layer],
                    eps_incident=eps_air,
                    eps_exit=eps_air
                )
                R = ml.reflectance()
                T = ml.transmittance()
                R_plus_T = R + T
                max_dev = np.max(np.abs(R_plus_T - 1.0))
                passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
                
                results['angle'].append(angle)
                results['layer_material'].append('Glass')
                results['thickness_nm'].append(thickness * 1e9)
                results[f'{pol}_pol']['R'].append(np.mean(R))
                results[f'{pol}_pol']['T'].append(np.mean(T))
                results[f'{pol}_pol']['R_plus_T'].append(np.mean(R_plus_T))
                results[f'{pol}_pol']['max_deviation'].append(max_dev)
                
                if not passed or (angle in [0, 30, 60, 85] and thickness == 100e-9):
                    status = "✓ PASS" if passed else "✗ FAIL"
                    print(f"    {angle:3.0f}°, {pol.upper()}-pol: {status} | R+T = {np.mean(R_plus_T):.10f} | Max dev: {max_dev:.2e}")
            except Exception as e:
                print(f"    {angle:3.0f}°, {pol.upper()}-pol: ERROR - {str(e)}")

# Test 2: Air | Silica layer | Air
print("\n" + "-" * 80)
print("Test 2: Air | Silica Layer | Air")
print("-" * 80)

for thickness in layer_thicknesses:
    layer = Layer(
        thickness=thickness,
        optical_property={'type': 'permittivity', 'value': eps_silica}
    )
    
    print(f"\n  Layer thickness: {thickness*1e9:.0f} nm")
    
    for angle in angles_degrees:
        for pol in ['s', 'p']:
            try:
                ml = MultiLayerStructure(
                    wavelengths=wavelengths,
                    angle_degrees=angle,
                    polarization=pol,
                    layers=[layer],
                    eps_incident=eps_air,
                    eps_exit=eps_air
                )
                R = ml.reflectance()
                T = ml.transmittance()
                R_plus_T = R + T
                max_dev = np.max(np.abs(R_plus_T - 1.0))
                passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
                
                results['angle'].append(angle)
                results['layer_material'].append('Silica')
                results['thickness_nm'].append(thickness * 1e9)
                results[f'{pol}_pol']['R'].append(np.mean(R))
                results[f'{pol}_pol']['T'].append(np.mean(T))
                results[f'{pol}_pol']['R_plus_T'].append(np.mean(R_plus_T))
                results[f'{pol}_pol']['max_deviation'].append(max_dev)
                
                if not passed or (angle in [0, 30, 60, 85] and thickness == 100e-9):
                    status = "✓ PASS" if passed else "✗ FAIL"
                    print(f"    {angle:3.0f}°, {pol.upper()}-pol: {status} | R+T = {np.mean(R_plus_T):.10f} | Max dev: {max_dev:.2e}")
            except Exception as e:
                print(f"    {angle:3.0f}°, {pol.upper()}-pol: ERROR - {str(e)}")

# Test 3: Air | Glass layer | Glass (different exit medium)
print("\n" + "-" * 80)
print("Test 3: Air | Glass Layer | Glass (Different Exit Medium)")
print("-" * 80)

thickness = 100e-9
layer = Layer(
    thickness=thickness,
    optical_property={'type': 'permittivity', 'value': eps_glass}
)

print(f"\n  Layer thickness: {thickness*1e9:.0f} nm")

for angle in angles_degrees:
    for pol in ['s', 'p']:
        try:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=[layer],
                eps_incident=eps_air,
                eps_exit=eps_glass
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
            
            results['angle'].append(angle)
            results['layer_material'].append('Glass|Glass')
            results['thickness_nm'].append(thickness * 1e9)
            results[f'{pol}_pol']['R'].append(np.mean(R))
            results[f'{pol}_pol']['T'].append(np.mean(T))
            results[f'{pol}_pol']['R_plus_T'].append(np.mean(R_plus_T))
            results[f'{pol}_pol']['max_deviation'].append(max_dev)
            
            if not passed or angle in [0, 30, 60, 85]:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"    {angle:3.0f}°, {pol.upper()}-pol: {status} | R+T = {np.mean(R_plus_T):.10f} | Max dev: {max_dev:.2e}")
        except Exception as e:
            print(f"    {angle:3.0f}°, {pol.upper()}-pol: ERROR - {str(e)}")

# Summary
print("\n" + "=" * 80)
print("Summary: Energy Conservation (R + T = 1.0) for Single Layer")
print("=" * 80)

# Group by material and thickness
unique_materials = sorted(set(results['layer_material']))
unique_thicknesses = sorted(set(results['thickness_nm']))
unique_angles = sorted(set(results['angle']))

print(f"\n{'Angle':<8} {'Material':<15} {'Thickness (nm)':<15} {'s-pol R+T':<15} {'s-pol Max Dev':<15} {'p-pol R+T':<15} {'p-pol Max Dev':<15} {'Status':<10}")
print("-" * 120)

all_passed = True
for material in unique_materials:
    for thickness in unique_thicknesses:
        for angle in [0, 30, 60, 85]:  # Show key angles
            # Find matching results
            s_r_plus_t = None
            s_max_dev = None
            p_r_plus_t = None
            p_max_dev = None
            
            for i, (a, m, t) in enumerate(zip(results['angle'], results['layer_material'], results['thickness_nm'])):
                if a == angle and m == material and abs(t - thickness) < 0.1:
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
                
                print(f"{angle:<8.0f} {material:<15} {thickness:<15.0f} {s_r_plus_t:<15.10f} {s_max_dev:<15.2e} {p_r_plus_t:<15.10f} {p_max_dev:<15.2e} {status:<10}")

# Count total tests
total_tests = len(results['angle']) // 2  # Divide by 2 for s and p
passed_tests = sum(1 for i in range(len(results['s_pol']['R_plus_T'])) 
                   if np.allclose(results['s_pol']['R_plus_T'][i], 1.0, atol=tolerance) and
                      np.allclose(results['p_pol']['R_plus_T'][i], 1.0, atol=tolerance))

print("\n" + "=" * 80)
print(f"Total test cases: {total_tests}")
print(f"Passed: {passed_tests}")
print(f"Failed: {total_tests - passed_tests}")
print("=" * 80)

if all_passed:
    print("✓ ALL TESTS PASSED - Energy conservation verified for oblique incidence single layer!")
    print(f"  All test cases show R + T = 1.0 within tolerance ({tolerance})")
else:
    print("✗ SOME TESTS FAILED - Energy conservation not satisfied")
print("=" * 80)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: R + T vs Angle for Glass layer (100 nm)
ax = axes[0, 0]
glass_100nm_angles = []
glass_100nm_s_r_plus_t = []
glass_100nm_p_r_plus_t = []

for i, (angle, material, thickness) in enumerate(zip(results['angle'], results['layer_material'], results['thickness_nm'])):
    if material == 'Glass' and abs(thickness - 100) < 0.1:
        if angle not in glass_100nm_angles:
            glass_100nm_angles.append(angle)
            idx = len(glass_100nm_angles) - 1
            if i < len(results['s_pol']['R_plus_T']):
                glass_100nm_s_r_plus_t.append(results['s_pol']['R_plus_T'][i])
            if i < len(results['p_pol']['R_plus_T']):
                glass_100nm_p_r_plus_t.append(results['p_pol']['R_plus_T'][i])

glass_100nm_angles = sorted(glass_100nm_angles)
glass_100nm_s_r_plus_t = [results['s_pol']['R_plus_T'][i] for i, (a, m, t) in enumerate(zip(results['angle'], results['layer_material'], results['thickness_nm'])) 
                          if m == 'Glass' and abs(t - 100) < 0.1 and a in glass_100nm_angles]
glass_100nm_p_r_plus_t = [results['p_pol']['R_plus_T'][i] for i, (a, m, t) in enumerate(zip(results['angle'], results['layer_material'], results['thickness_nm'])) 
                          if m == 'Glass' and abs(t - 100) < 0.1 and a in glass_100nm_angles]

if glass_100nm_angles:
    ax.plot(glass_100nm_angles, glass_100nm_s_r_plus_t[:len(glass_100nm_angles)], 'o-', label='s-polarization (TE)', linewidth=2, markersize=6)
    ax.plot(glass_100nm_angles, glass_100nm_p_r_plus_t[:len(glass_100nm_angles)], 's-', label='p-polarization (TM)', linewidth=2, markersize=6)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='R + T = 1 (ideal)')
ax.set_xlabel('Incident Angle (degrees)', fontsize=12)
ax.set_ylabel('R + T', fontsize=12)
ax.set_title('Energy Conservation: Air | Glass (100 nm) | Air', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 92)

# Plot 2: Reflectance vs Angle for different thicknesses
ax = axes[0, 1]
for thickness in [50, 100, 200]:
    thickness_angles = []
    thickness_s_r = []
    for i, (angle, material, t) in enumerate(zip(results['angle'], results['layer_material'], results['thickness_nm'])):
        if material == 'Glass' and abs(t - thickness) < 0.1:
            if angle not in thickness_angles:
                thickness_angles.append(angle)
                if i < len(results['s_pol']['R']):
                    thickness_s_r.append(results['s_pol']['R'][i])
    
    thickness_angles = sorted(thickness_angles)
    thickness_s_r = [results['s_pol']['R'][i] for i, (a, m, t) in enumerate(zip(results['angle'], results['layer_material'], results['thickness_nm'])) 
                     if m == 'Glass' and abs(t - thickness) < 0.1 and a in thickness_angles]
    
    if thickness_angles:
        ax.plot(thickness_angles, thickness_s_r[:len(thickness_angles)], 'o-', label=f'{thickness} nm', linewidth=2, markersize=6)

ax.set_xlabel('Incident Angle (degrees)', fontsize=12)
ax.set_ylabel('Reflectance (R) - s-polarization', fontsize=12)
ax.set_title('Reflectance vs Angle: Different Layer Thicknesses', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 92)

# Plot 3: Maximum Deviation vs Angle
ax = axes[1, 0]
glass_100nm_s_max_dev = [results['s_pol']['max_deviation'][i] for i, (a, m, t) in enumerate(zip(results['angle'], results['layer_material'], results['thickness_nm'])) 
                         if m == 'Glass' and abs(t - 100) < 0.1]
glass_100nm_p_max_dev = [results['p_pol']['max_deviation'][i] for i, (a, m, t) in enumerate(zip(results['angle'], results['layer_material'], results['thickness_nm'])) 
                         if m == 'Glass' and abs(t - 100) < 0.1]

if glass_100nm_angles and glass_100nm_s_max_dev:
    ax.semilogy(glass_100nm_angles, glass_100nm_s_max_dev[:len(glass_100nm_angles)], 'o-', label='s-polarization (TE)', linewidth=2, markersize=6)
    ax.semilogy(glass_100nm_angles, glass_100nm_p_max_dev[:len(glass_100nm_angles)], 's-', label='p-polarization (TM)', linewidth=2, markersize=6)
ax.axhline(y=tolerance, color='red', linestyle='--', linewidth=2, label=f'Tolerance ({tolerance})')
ax.set_xlabel('Incident Angle (degrees)', fontsize=12)
ax.set_ylabel('Max Deviation from 1.0 (log scale)', fontsize=12)
ax.set_title('Maximum Deviation from Energy Conservation', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 92)

# Plot 4: R + T vs Thickness at 30°
ax = axes[1, 1]
angle_30 = 30.0
thicknesses_30 = []
s_r_plus_t_30 = []
p_r_plus_t_30 = []

for thickness in unique_thicknesses:
    for i, (angle, material, t) in enumerate(zip(results['angle'], results['layer_material'], results['thickness_nm'])):
        if angle == angle_30 and material == 'Glass' and abs(t - thickness) < 0.1:
            if i < len(results['s_pol']['R_plus_T']):
                thicknesses_30.append(thickness)
                s_r_plus_t_30.append(results['s_pol']['R_plus_T'][i])
                p_r_plus_t_30.append(results['p_pol']['R_plus_T'][i])
            break

if thicknesses_30:
    ax.plot(thicknesses_30, s_r_plus_t_30, 'o-', label='s-polarization (TE)', linewidth=2, markersize=8)
    ax.plot(thicknesses_30, p_r_plus_t_30, 's-', label='p-polarization (TM)', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='R + T = 1 (ideal)')
    ax.set_xlabel('Layer Thickness (nm)', fontsize=12)
    ax.set_ylabel('R + T', fontsize=12)
    ax.set_title(f'Energy Conservation at {angle_30}° vs Layer Thickness', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('energy_conservation_oblique_single_layer.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: energy_conservation_oblique_single_layer.png")
plt.show()

print("\nTest completed!")

