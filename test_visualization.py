#!/usr/bin/env python
"""
Test 15: Visualization

Generates visualization plots for energy conservation tests.
Requires matplotlib to be installed.
"""

import numpy as np
from TMatrix import Layer, MultiLayerStructure

# Try to import matplotlib for visualization (optional)
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, visualization will be skipped")
    exit(0)

from test_common_setup import (
    tolerance,
    wavelengths,
    eps_air_const,
    eps_silica_const,
    eps_titanium_const,
)

print("=" * 80)
print("Test 15: Visualization")
print("=" * 80)
print("  Generating plots...")

# Collect data for visualization
viz_layers = list(range(0, 11))
viz_angles = [0.0, 30.0, 45.0, 60.0, 85.0]
viz_data = (
    {}
)  # {(n_layers, angle): {"s": (r_plus_t, max_dev), "p": (r_plus_t, max_dev)}}

for n_layers in viz_layers:
    layers = []
    for i in range(n_layers):
        thickness = 50e-9 + (i * 5e-9)
        if i % 2 == 0:
            layers.append(
                Layer(
                    thickness=thickness,
                    optical_property={
                        "type": "permittivity",
                        "value": eps_silica_const,
                    },
                )
            )
        else:
            layers.append(
                Layer(
                    thickness=thickness,
                    optical_property={
                        "type": "permittivity",
                        "value": eps_titanium_const,
                    },
                )
            )

    for angle in viz_angles:
        key = (n_layers, angle)
        viz_data[key] = {}
        for pol in ["s", "p"]:
            ml = MultiLayerStructure(
                wavelengths=wavelengths,
                angle_degrees=angle,
                polarization=pol,
                layers=layers,
                eps_incident=eps_air_const,
                eps_exit=eps_air_const,
            )
            R = ml.reflectance()
            T = ml.transmittance()
            R_plus_T = R + T
            max_dev = np.max(np.abs(R_plus_T - 1.0))
            viz_data[key][pol] = (np.mean(R_plus_T), max_dev)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: R + T vs Number of Layers (normal incidence)
ax = axes[0, 0]
n_layers_normal = sorted([n for n, a in viz_data.keys() if a == 0.0])
s_r_plus_t_normal = [viz_data[(n, 0.0)]["s"][0] for n in n_layers_normal]
p_r_plus_t_normal = [viz_data[(n, 0.0)]["p"][0] for n in n_layers_normal]
ax.plot(
    n_layers_normal,
    s_r_plus_t_normal,
    "o-",
    label="s-polarization (TE)",
    linewidth=2,
    markersize=6,
)
ax.plot(
    n_layers_normal,
    p_r_plus_t_normal,
    "s-",
    label="p-polarization (TM)",
    linewidth=2,
    markersize=6,
)
ax.axhline(
    y=1.0, color="red", linestyle="--", linewidth=2, label="R + T = 1 (ideal)"
)
ax.set_xlabel("Number of Layers", fontsize=12)
ax.set_ylabel("R + T", fontsize=12)
ax.set_title(
    "Energy Conservation: R + T vs Number of Layers (Normal Incidence)",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: Maximum Deviation vs Number of Layers
ax = axes[0, 1]
s_max_dev_normal = [viz_data[(n, 0.0)]["s"][1] for n in n_layers_normal]
p_max_dev_normal = [viz_data[(n, 0.0)]["p"][1] for n in n_layers_normal]
ax.semilogy(
    n_layers_normal,
    s_max_dev_normal,
    "o-",
    label="s-polarization (TE)",
    linewidth=2,
    markersize=6,
)
ax.semilogy(
    n_layers_normal,
    p_max_dev_normal,
    "s-",
    label="p-polarization (TM)",
    linewidth=2,
    markersize=6,
)
ax.axhline(
    y=tolerance,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Tolerance ({tolerance})",
)
ax.set_xlabel("Number of Layers", fontsize=12)
ax.set_ylabel("Max Deviation from 1.0 (log scale)", fontsize=12)
ax.set_title(
    "Maximum Deviation from Energy Conservation", fontsize=14, fontweight="bold"
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 3: R + T vs Angle (single interface)
ax = axes[1, 0]
angles_interface_viz = sorted([a for n, a in viz_data.keys() if n == 0])
s_r_plus_t_interface = [viz_data[(0, a)]["s"][0] for a in angles_interface_viz]
p_r_plus_t_interface = [viz_data[(0, a)]["p"][0] for a in angles_interface_viz]
ax.plot(
    angles_interface_viz,
    s_r_plus_t_interface,
    "o-",
    label="s-polarization (TE)",
    linewidth=2,
    markersize=6,
)
ax.plot(
    angles_interface_viz,
    p_r_plus_t_interface,
    "s-",
    label="p-polarization (TM)",
    linewidth=2,
    markersize=6,
)
ax.axhline(
    y=1.0, color="red", linestyle="--", linewidth=2, label="R + T = 1 (ideal)"
)
ax.set_xlabel("Incident Angle (degrees)", fontsize=12)
ax.set_ylabel("R + T", fontsize=12)
ax.set_title(
    "Energy Conservation: R + T vs Angle (Single Interface)",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 4: Comparison at 45° for different layer counts
ax = axes[1, 1]
n_layers_45 = sorted([n for n, a in viz_data.keys() if a == 45.0])
s_r_plus_t_45 = [viz_data[(n, 45.0)]["s"][0] for n in n_layers_45]
p_r_plus_t_45 = [viz_data[(n, 45.0)]["p"][0] for n in n_layers_45]
ax.plot(
    n_layers_45,
    s_r_plus_t_45,
    "o-",
    label="s-polarization (TE)",
    linewidth=2,
    markersize=6,
)
ax.plot(
    n_layers_45,
    p_r_plus_t_45,
    "s-",
    label="p-polarization (TM)",
    linewidth=2,
    markersize=6,
)
ax.axhline(
    y=1.0, color="red", linestyle="--", linewidth=2, label="R + T = 1 (ideal)"
)
ax.set_xlabel("Number of Layers", fontsize=12)
ax.set_ylabel("R + T", fontsize=12)
ax.set_title(
    "Energy Conservation at 45° vs Number of Layers", fontsize=14, fontweight="bold"
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("energy_conservation_comprehensive.png", dpi=300, bbox_inches="tight")
print("  ✓ Plots saved to: energy_conservation_comprehensive.png")
# Don't show plots automatically (comment out plt.show() to avoid blocking)
# plt.show()

print("\n" + "=" * 80)

