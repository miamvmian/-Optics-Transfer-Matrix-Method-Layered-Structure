#!/usr/bin/env python
"""
Energy Conservation Test for Normal Incidence with Different Numbers of Layers

Tests R + T = 1.0 for lossless structures at normal incidence (0°)
with varying numbers of layers (0 to 10 layers).
"""

import numpy as np
import matplotlib.pyplot as plt
from TMatrix import Layer, MultiLayerStructure

print("=" * 80)
print("Energy Conservation Test: Normal Incidence, Different Numbers of Layers")
print("=" * 80)

# Test parameters
tolerance = 1e-10
wavelengths = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
angle_degrees = 0.0  # Normal incidence

# Material properties (lossless - real permittivities)
eps_air = 1.0
eps_glass = 2.25  # n = 1.5
eps_silica = 2.13  # n ≈ 1.46
eps_titanium = 5.76  # n ≈ 2.4

print(f"\nTest Configuration:")
print(f"  Wavelengths: {', '.join([f'{w*1e9:.0f}' for w in wavelengths])} nm")
print(f"  Incident angle: {angle_degrees}° (normal incidence)")
print(f"  Tolerance: {tolerance}")
print(f"  Materials: Air (ε=1.0), Glass (ε=2.25), Silica (ε=2.13), Titanium (ε=5.76)")

# Store results for plotting
results = {
    'n_layers': [],
    's_pol': {'R': [], 'T': [], 'R_plus_T': [], 'max_deviation': []},
    'p_pol': {'R': [], 'T': [], 'R_plus_T': [], 'max_deviation': []}
}

# Test 1: Single interface (0 layers)
print("\n" + "-" * 80)
print("Test 1: Single Interface (0 layers) - Air | Glass")
print("-" * 80)

for pol in ['s', 'p']:
    ml = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=angle_degrees,
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
    
    results['n_layers'].append(0)
    results[f'{pol}_pol']['R'].append(np.mean(R))
    results[f'{pol}_pol']['T'].append(np.mean(T))
    results[f'{pol}_pol']['R_plus_T'].append(np.mean(R_plus_T))
    results[f'{pol}_pol']['max_deviation'].append(max_dev)
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {pol.upper()}-polarization: {status}")
    print(f"    Mean R: {np.mean(R):.10f}")
    print(f"    Mean T: {np.mean(T):.10f}")
    print(f"    Mean R+T: {np.mean(R_plus_T):.10f}")
    print(f"    Max deviation from 1.0: {max_dev:.2e}")

# Test 2: Multiple layers (1 to 10 layers)
print("\n" + "-" * 80)
print("Test 2: Multiple Layers (1 to 10) - Air | [Alternating Silica/Titanium] | Air")
print("-" * 80)

for n_layers in range(1, 11):
    # Create alternating layers
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 5e-9)  # Varying thicknesses
        if i % 2 == 0:
            layers.append(Layer(
                thickness=thickness,
                optical_property={'type': 'permittivity', 'value': eps_silica}
            ))
        else:
            layers.append(Layer(
                thickness=thickness,
                optical_property={'type': 'permittivity', 'value': eps_titanium}
            ))
    
    print(f"\n  {n_layers} layer(s):")
    
    for pol in ['s', 'p']:
        ml = MultiLayerStructure(
            wavelengths=wavelengths,
            angle_degrees=angle_degrees,
            polarization=pol,
            layers=layers,
            eps_incident=eps_air,
            eps_exit=eps_air
        )
        R = ml.reflectance()
        T = ml.transmittance()
        R_plus_T = R + T
        max_dev = np.max(np.abs(R_plus_T - 1.0))
        passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
        
        results['n_layers'].append(n_layers)
        results[f'{pol}_pol']['R'].append(np.mean(R))
        results[f'{pol}_pol']['T'].append(np.mean(T))
        results[f'{pol}_pol']['R_plus_T'].append(np.mean(R_plus_T))
        results[f'{pol}_pol']['max_deviation'].append(max_dev)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {pol.upper()}-pol: {status} | R+T = {np.mean(R_plus_T):.10f} | Max dev: {max_dev:.2e}")

# Test 3: Different exit medium
print("\n" + "-" * 80)
print("Test 3: 2 Layers with Glass Exit Medium")
print("-" * 80)

layers = [
    Layer(thickness=100e-9, optical_property={'type': 'permittivity', 'value': eps_silica}),
    Layer(thickness=100e-9, optical_property={'type': 'permittivity', 'value': eps_titanium})
]

for pol in ['s', 'p']:
    ml = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=angle_degrees,
        polarization=pol,
        layers=layers,
        eps_incident=eps_air,
        eps_exit=eps_glass
    )
    R = ml.reflectance()
    T = ml.transmittance()
    R_plus_T = R + T
    max_dev = np.max(np.abs(R_plus_T - 1.0))
    passed = np.allclose(R_plus_T, 1.0, atol=tolerance)
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {pol.upper()}-polarization: {status}")
    print(f"    Mean R: {np.mean(R):.10f}")
    print(f"    Mean T: {np.mean(T):.10f}")
    print(f"    Mean R+T: {np.mean(R_plus_T):.10f}")
    print(f"    Max deviation from 1.0: {max_dev:.2e}")

# Summary
print("\n" + "=" * 80)
print("Summary: Energy Conservation (R + T = 1.0)")
print("=" * 80)

# Extract unique layer counts
unique_layers = sorted(set(results['n_layers']))

print(f"\n{'Layers':<10} {'s-pol R+T':<15} {'s-pol Max Dev':<15} {'p-pol R+T':<15} {'p-pol Max Dev':<15} {'Status':<10}")
print("-" * 80)

all_passed = True
for n in unique_layers:
    # Find indices for this layer count
    indices = [i for i, n_lay in enumerate(results['n_layers']) if n_lay == n]
    
    if len(indices) >= 2:
        # Get s and p results
        s_idx = indices[0] if results['n_layers'][indices[0]] == n else None
        p_idx = indices[1] if len(indices) > 1 else None
        
        # Find correct indices
        s_values = []
        p_values = []
        for idx in indices:
            if idx < len(results['s_pol']['R_plus_T']):
                s_values.append((results['s_pol']['R_plus_T'][idx], results['s_pol']['max_deviation'][idx]))
            if idx < len(results['p_pol']['R_plus_T']):
                p_values.append((results['p_pol']['R_plus_T'][idx], results['p_pol']['max_deviation'][idx]))
        
        if s_values and p_values:
            s_r_plus_t, s_max_dev = s_values[0]
            p_r_plus_t, p_max_dev = p_values[0] if p_values else (None, None)
            
            s_passed = np.allclose(s_r_plus_t, 1.0, atol=tolerance)
            p_passed = np.allclose(p_r_plus_t, 1.0, atol=tolerance) if p_r_plus_t is not None else False
            
            status = "✓ PASS" if (s_passed and p_passed) else "✗ FAIL"
            if not (s_passed and p_passed):
                all_passed = False
            
            print(f"{n:<10} {s_r_plus_t:<15.10f} {s_max_dev:<15.2e} {p_r_plus_t:<15.10f} {p_max_dev:<15.2e} {status:<10}")

print("\n" + "=" * 80)
if all_passed:
    print("✓ ALL TESTS PASSED - Energy conservation verified for normal incidence!")
    print(f"  All test cases show R + T = 1.0 within tolerance ({tolerance})")
else:
    print("✗ SOME TESTS FAILED - Energy conservation not satisfied")
print("=" * 80)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: R + T vs Number of Layers
ax = axes[0, 0]
unique_layers_sorted = sorted(set(results['n_layers']))
s_r_plus_t_values = []
p_r_plus_t_values = []
for n in unique_layers_sorted:
    indices = [i for i, n_lay in enumerate(results['n_layers']) if n_lay == n]
    if indices:
        s_val = results['s_pol']['R_plus_T'][indices[0]] if indices[0] < len(results['s_pol']['R_plus_T']) else None
        p_val = results['p_pol']['R_plus_T'][indices[0]] if indices[0] < len(results['p_pol']['R_plus_T']) else None
        s_r_plus_t_values.append(s_val if s_val is not None else np.nan)
        p_r_plus_t_values.append(p_val if p_val is not None else np.nan)
    else:
        s_r_plus_t_values.append(np.nan)
        p_r_plus_t_values.append(np.nan)

ax.plot(unique_layers_sorted, s_r_plus_t_values, 'o-', label='s-polarization (TE)', linewidth=2, markersize=6)
ax.plot(unique_layers_sorted, p_r_plus_t_values, 's-', label='p-polarization (TM)', linewidth=2, markersize=6)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='R + T = 1 (ideal)')
ax.set_xlabel('Number of Layers', fontsize=12)
ax.set_ylabel('R + T', fontsize=12)
ax.set_title('Energy Conservation: R + T vs Number of Layers', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, max(unique_layers_sorted) + 0.5)

# Plot 2: Maximum Deviation vs Number of Layers
ax = axes[0, 1]
s_max_dev_values = []
p_max_dev_values = []
for n in unique_layers_sorted:
    indices = [i for i, n_lay in enumerate(results['n_layers']) if n_lay == n]
    if indices:
        s_val = results['s_pol']['max_deviation'][indices[0]] if indices[0] < len(results['s_pol']['max_deviation']) else None
        p_val = results['p_pol']['max_deviation'][indices[0]] if indices[0] < len(results['p_pol']['max_deviation']) else None
        s_max_dev_values.append(s_val if s_val is not None else np.nan)
        p_max_dev_values.append(p_val if p_val is not None else np.nan)
    else:
        s_max_dev_values.append(np.nan)
        p_max_dev_values.append(np.nan)

ax.semilogy(unique_layers_sorted, s_max_dev_values, 'o-', label='s-polarization (TE)', linewidth=2, markersize=6)
ax.semilogy(unique_layers_sorted, p_max_dev_values, 's-', label='p-polarization (TM)', linewidth=2, markersize=6)
ax.axhline(y=tolerance, color='red', linestyle='--', linewidth=2, label=f'Tolerance ({tolerance})')
ax.set_xlabel('Number of Layers', fontsize=12)
ax.set_ylabel('Max Deviation from 1.0 (log scale)', fontsize=12)
ax.set_title('Maximum Deviation from Energy Conservation', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, max(unique_layers_sorted) + 0.5)

# Plot 3: Reflectance vs Number of Layers
ax = axes[1, 0]
s_r_values = []
p_r_values = []
for n in unique_layers_sorted:
    indices = [i for i, n_lay in enumerate(results['n_layers']) if n_lay == n]
    if indices:
        s_val = results['s_pol']['R'][indices[0]] if indices[0] < len(results['s_pol']['R']) else None
        p_val = results['p_pol']['R'][indices[0]] if indices[0] < len(results['p_pol']['R']) else None
        s_r_values.append(s_val if s_val is not None else np.nan)
        p_r_values.append(p_val if p_val is not None else np.nan)
    else:
        s_r_values.append(np.nan)
        p_r_values.append(np.nan)

ax.plot(unique_layers_sorted, s_r_values, 'o-', label='s-polarization (TE)', linewidth=2, markersize=6)
ax.plot(unique_layers_sorted, p_r_values, 's-', label='p-polarization (TM)', linewidth=2, markersize=6)
ax.set_xlabel('Number of Layers', fontsize=12)
ax.set_ylabel('Reflectance (R)', fontsize=12)
ax.set_title('Reflectance vs Number of Layers', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, max(unique_layers_sorted) + 0.5)

# Plot 4: Transmittance vs Number of Layers
ax = axes[1, 1]
s_t_values = []
p_t_values = []
for n in unique_layers_sorted:
    indices = [i for i, n_lay in enumerate(results['n_layers']) if n_lay == n]
    if indices:
        s_val = results['s_pol']['T'][indices[0]] if indices[0] < len(results['s_pol']['T']) else None
        p_val = results['p_pol']['T'][indices[0]] if indices[0] < len(results['p_pol']['T']) else None
        s_t_values.append(s_val if s_val is not None else np.nan)
        p_t_values.append(p_val if p_val is not None else np.nan)
    else:
        s_t_values.append(np.nan)
        p_t_values.append(np.nan)

ax.plot(unique_layers_sorted, s_t_values, 'o-', label='s-polarization (TE)', linewidth=2, markersize=6)
ax.plot(unique_layers_sorted, p_t_values, 's-', label='p-polarization (TM)', linewidth=2, markersize=6)
ax.set_xlabel('Number of Layers', fontsize=12)
ax.set_ylabel('Transmittance (T)', fontsize=12)
ax.set_title('Transmittance vs Number of Layers', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, max(unique_layers_sorted) + 0.5)

plt.tight_layout()
plt.savefig('energy_conservation_normal_incidence.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: energy_conservation_normal_incidence.png")
plt.show()

print("\nTest completed!")

