#!/usr/bin/env python
"""
Test: Field Visualization

Tests the field mesh grid visualization functions in TMatrix_Field.py.
Generates field distribution plots for multi-layer structures.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
from TMatrix import Layer, MultiLayerStructure
from TMatrix_Field import (
    visualize_field_mesh,
    visualize_field_layers_separate,
    create_field_mesh_grid,
    calculate_field_on_mesh,
    calculate_layer_field_data,
)

# Try to import matplotlib for visualization (optional)
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, visualization will be skipped")
    exit(0)

from test_common_setup import (
    eps_air_const,
    eps_silica_const,
    eps_Nb2O5_const,
    E_in,
)

# Ensure plots directory exists
plots_dir = Path(__file__).parent / "plots"
plots_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Test: Field Visualization")
print("=" * 80)

# Test 1: Simple single layer structure
print("\n" + "-" * 80)
print("Test 1: Single Layer Structure (Air | SiO2 | Air)")
print("-" * 80)

wavelengths_single = np.array([1500e-9])  # Single wavelength: 1500 nm for single layer
angle_degrees = 0.0  # Normal incidence

# Quarter-wave plate thickness: d = λ/(4*n) = λ/(4*sqrt(eps))
# For λ = 1500 nm, antireflection coating condition
quarter_wave_thickness = (1 / 4 * 1500e-9) / np.sqrt(1.23**2)

layers_simple = [
    Layer(
        thickness=quarter_wave_thickness,
        optical_property={"type": "permittivity", "value": 1.23**2},
    )
]

structure_simple = MultiLayerStructure(
    wavelengths=wavelengths_single,
    angle_degrees=angle_degrees,
    polarization="s",
    layers=layers_simple,
    eps_incident=eps_air_const,
    eps_exit=1.52**2,
)

print(f"Structure: Air | SiO2 (quarter-wave plate@1500nm) | Air")
print(f"Thickness: {quarter_wave_thickness*1e9:.2f} nm")
print(f"Wavelength: {wavelengths_single[0]*1e9:.1f} nm")
print(f"Angle: {angle_degrees}°")
print(f"Polarization: s (TE)")

# Calculate reflectance and transmittance
R_s = structure_simple.reflectance()[0]
T_s = structure_simple.transmittance()[0]
print(f"\nReflectance: R = {R_s:.6f}")
print(f"Transmittance: T = {T_s:.6f}")
print(f"R + T = {R_s + T_s:.6f}")

# Calculate layer field data
layer_fields = calculate_layer_field_data(structure_simple, E_in=E_in)
print(f"\nNumber of layers: {len(layer_fields)}")
for i, lf in enumerate(layer_fields):
    print(f"  Layer {i}: z = [{lf.z_start*1e9:.1f}, {lf.z_end*1e9:.1f}] nm")

# Visualize
print(f"\nUsing input electric field: E_in = {E_in:.6e} V/m")
print("Generating visualization plots...")
visualize_field_mesh(
    structure_simple,
    x_range=(-500e-9, 500e-9),
    n_x=100,
    n_z_per_layer=50,
    wavelength_index=0,
    field_type="normE",
    save_path=str((Path(__file__).parent / "plots" / "field_single_layer.png")),
    show_plot=False,
    E_in=E_in,
)

# Test 2: Multi-layer structure (Bragg mirror)
print("\n" + "-" * 80)
print("Test 2: Multi-Layer Structure (Bragg Mirror)")
print("-" * 80)

wavelengths_bragg = np.array(
    [1300e-9, 1400e-9, 1500e-9, 1700e-9]
)  # Multiple wavelengths: [1300, 1400, 1500, 1700] nm for multilayer
layers_bragg = []

# Create 10 periods of Nb2O5 | SiO2 (20 layers total)
for _ in range(10):
    layers_bragg.append(
        Layer(
            thickness=159e-9,
            optical_property={"type": "permittivity", "value": eps_Nb2O5_const},
        )
    )
    layers_bragg.append(
        Layer(
            thickness=246e-9,
            optical_property={"type": "permittivity", "value": eps_silica_const},
        )
    )

structure_bragg = MultiLayerStructure(
    wavelengths=wavelengths_bragg,
    angle_degrees=0.0,
    polarization="s",
    layers=layers_bragg,
    eps_incident=eps_air_const,
    eps_exit=eps_silica_const,
)

print(f"Structure: Air | [Nb2O5 | SiO2]10 | SiO2")
print(f"Wavelengths: {wavelengths_bragg*1e9} nm")
print(f"Angle: 0.0°")
print(f"Polarization: s (TE)")
print(f"Total layers: {len(layers_bragg)}")

# Calculate reflectance and transmittance
R = structure_bragg.reflectance()
T = structure_bragg.transmittance()
print(f"\nReflectance: R = {R}")
print(f"Transmittance: T = {T}")
print(f"R + T = {R + T}")

# Visualize combined view for each wavelength
print("\nGenerating combined field visualizations...")
for wl_idx, wl in enumerate(wavelengths_bragg):
    print(f"  Visualizing wavelength {wl_idx}: {wl*1e9:.1f} nm")
    visualize_field_mesh(
        structure_bragg,
        x_range=(-500e-9, 500e-9),
        n_x=100,
        n_z_per_layer=30,
        wavelength_index=wl_idx,
        field_type="normE",
        save_path=str(
            Path(__file__).parent
            / "plots"
            / f"field_bragg_combined_wl_{wl_idx}_{int(wl*1e9)}nm.png"
        ),
        show_plot=False,
        E_in=E_in,
    )

# Visualize layers separately for first wavelength
print("Generating separate layer visualizations...")
visualize_field_layers_separate(
    structure_bragg,
    x_range=(-500e-9, 500e-9),
    n_x=100,
    n_z_per_layer=30,
    wavelength_index=0,
    field_type="normE",
    save_path=str(Path(__file__).parent / "plots" / "field_bragg_layers.png"),
    show_plot=False,
    E_in=E_in,
)

# Test 3: Test different field types
print("\n" + "-" * 80)
print("Test 3: Different Field Types")
print("-" * 80)

structure_test = structure_simple  # Use simple structure

for field_type in ["normE", "total", "forward", "backward"]:
    print(f"\nTesting field type: {field_type}")

    # Create mesh grid
    mesh_data = create_field_mesh_grid(
        structure_test,
        x_range=(-500e-9, 500e-9),
        n_x=50,
        n_z_per_layer=30,
        wavelength_index=0,
        E_in=E_in,
    )

    # Calculate field
    field_data = calculate_field_on_mesh(mesh_data, field_type)

    print(f"  Mesh created: x shape = {mesh_data['x'].shape}")
    print(f"  Number of layers: {len(field_data['field_values'])}")
    for i, fv in enumerate(field_data["field_values"]):
        print(f"    Layer {i}: field shape = {fv.shape}")

# Test 4: P-polarization
print("\n" + "-" * 80)
print("Test 4: P-Polarization (TM)")
print("-" * 80)

structure_p = MultiLayerStructure(
    wavelengths=wavelengths_single,
    angle_degrees=0.0,
    polarization="p",
    layers=layers_simple,
    eps_incident=eps_air_const,
    eps_exit=eps_air_const,
)

print(f"Structure: Air | SiO2 (quarter-wave plate@1500nm) | Air")
print(f"Thickness: {quarter_wave_thickness*1e9:.2f} nm")
print(f"Wavelength: {wavelengths_single[0]*1e9:.1f} nm")
print(f"Angle: 0.0°")
print(f"Polarization: p (TM)")

R_p = structure_p.reflectance()[0]
T_p = structure_p.transmittance()[0]
print(f"\nReflectance: R = {R_p:.6f}")
print(f"Transmittance: T = {T_p:.6f}")

# Visualize for first wavelength
visualize_field_mesh(
    structure_p,
    x_range=(-500e-9, 500e-9),
    n_x=100,
    n_z_per_layer=50,
    wavelength_index=0,
    field_type="normE",
    save_path=str(Path(__file__).parent / "plots" / "field_p_polarization.png"),
    show_plot=False,
    E_in=E_in,
)

# Test 5: Multiple wavelengths
print("\n" + "-" * 80)
print("Test 5: Multiple Wavelengths")
print("-" * 80)

wavelengths_multi = np.array([500e-9, 600e-9, 700e-9])
structure_multi = MultiLayerStructure(
    wavelengths=wavelengths_multi,
    angle_degrees=0.0,
    polarization="s",
    layers=layers_simple,
    eps_incident=eps_air_const,
    eps_exit=eps_air_const,
)

print(f"Structure: Air | SiO2 (quarter-wave plate@1500nm) | Air")
print(f"Thickness: {quarter_wave_thickness*1e9:.2f} nm (for 1500 nm)")
print(f"Wavelengths: {wavelengths_multi*1e9} nm")
print(f"Angle: 0.0°")
print(f"Polarization: s (TE)")

# Visualize for each wavelength
for wl_idx, wl in enumerate(wavelengths_multi):
    print(f"\nVisualizing wavelength {wl_idx}: {wl*1e9:.1f} nm")
    visualize_field_mesh(
        structure_multi,
        x_range=(-500e-9, 500e-9),
        n_x=100,
        n_z_per_layer=50,
        wavelength_index=wl_idx,
        field_type="normE",
        save_path=str(
            Path(__file__).parent
            / "plots"
            / f"field_multi_wl_{wl_idx}_{int(wl*1e9)}nm.png"
        ),
        show_plot=False,
        E_in=E_in,
    )

print("\n" + "=" * 80)
print("Field Visualization Tests Complete!")
print("=" * 80)
print("\nGenerated plots:")
print("  - field_single_layer.png (1500 nm)")
print(
    "  - field_bragg_combined_wl_*_*.png (for wavelengths: 1300, 1400, 1500, 1700 nm)"
)
print("  - field_bragg_layers.png (1300 nm)")
print("  - field_p_polarization.png (1500 nm)")
print("  - field_multi_wl_*.png")
