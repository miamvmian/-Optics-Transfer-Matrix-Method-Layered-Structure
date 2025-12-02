#!/usr/bin/env python
"""
Compare TMatrix results with COMSOL results and plot them on the same graph.
Analyzes TMatrix S-polarization vs COMSOL TE and TMatrix P-polarization vs COMSOL TM.
This version compares data at 0° incidence angle.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# File paths - read from data folder
data_dir = Path(__file__).parent / "data"
tmatrix_file = data_dir / "tmatrix_0DEG.txt"
comsol_te_file = data_dir / "comsol_TE_0DEG.txt"  # S-polarization (TE)
comsol_tm_file = data_dir / "comsol_TM_0DEG.txt"  # P-polarization (TM)

# Parse TMatrix data
tmatrix_wavelengths = []
tmatrix_R_s = []
tmatrix_R_p = []

with open(tmatrix_file, "r") as f:
    lines = f.readlines()
    # Find the data section (starts after "λ (nm)\tR_s\tR_p")
    data_started = False
    for line in lines:
        if "λ (nm)" in line and "R_s" in line:
            data_started = True
            continue
        if data_started and line.strip():
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    wl = float(parts[0])
                    r_s = float(parts[1])
                    r_p = float(parts[2])
                    tmatrix_wavelengths.append(wl)
                    tmatrix_R_s.append(r_s)
                    tmatrix_R_p.append(r_p)
                except ValueError:
                    continue

tmatrix_wavelengths = np.array(tmatrix_wavelengths)
tmatrix_R_s = np.array(tmatrix_R_s)
tmatrix_R_p = np.array(tmatrix_R_p)

# Parse COMSOL TE data (S-polarization)
comsol_te_wavelengths = []
comsol_te_reflectance = []

with open(comsol_te_file, "r") as f:
    lines = f.readlines()
    # Skip header lines (lines starting with %)
    for line in lines:
        if line.strip().startswith("%"):
            continue
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    # Format: freq(THz) PortName wavelength(nm) Reflectance
                    wavelength = float(parts[2])
                    reflectance = float(parts[3])
                    comsol_te_wavelengths.append(wavelength)
                    comsol_te_reflectance.append(reflectance)
                except (ValueError, IndexError):
                    continue

comsol_te_wavelengths = np.array(comsol_te_wavelengths)
comsol_te_reflectance = np.array(comsol_te_reflectance)

# Sort COMSOL TE data by wavelength
sort_idx = np.argsort(comsol_te_wavelengths)
comsol_te_wavelengths = comsol_te_wavelengths[sort_idx]
comsol_te_reflectance = comsol_te_reflectance[sort_idx]

# Parse COMSOL TM data (P-polarization)
comsol_tm_wavelengths = []
comsol_tm_reflectance = []

with open(comsol_tm_file, "r") as f:
    lines = f.readlines()
    # Skip header lines (lines starting with %)
    for line in lines:
        if line.strip().startswith("%"):
            continue
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    # Format: freq(THz) PortName wavelength(nm) Reflectance
                    wavelength = float(parts[2])
                    reflectance = float(parts[3])
                    comsol_tm_wavelengths.append(wavelength)
                    comsol_tm_reflectance.append(reflectance)
                except (ValueError, IndexError):
                    continue

comsol_tm_wavelengths = np.array(comsol_tm_wavelengths)
comsol_tm_reflectance = np.array(comsol_tm_reflectance)

# Sort COMSOL TM data by wavelength
sort_idx = np.argsort(comsol_tm_wavelengths)
comsol_tm_wavelengths = comsol_tm_wavelengths[sort_idx]
comsol_tm_reflectance = comsol_tm_reflectance[sort_idx]

# Create comparison plot with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: S-polarization (TE) comparison
ax1.plot(
    tmatrix_wavelengths,
    tmatrix_R_s,
    "b-",
    linewidth=2,
    label="TMatrix (S-polarization)",
    alpha=0.8,
)
ax1.plot(
    comsol_te_wavelengths,
    comsol_te_reflectance,
    "r--",
    linewidth=2,
    label="COMSOL (TE)",
    alpha=0.8,
    markersize=4,
)
ax1.set_ylabel("Reflectance", fontsize=12)
ax1.set_title(
    "S-polarization (TE) Comparison @ 0°: TMatrix vs COMSOL",
    fontsize=13,
    fontweight="bold",
)
ax1.legend(fontsize=11, loc="best")
ax1.grid(True, alpha=0.3)

# Plot 2: P-polarization (TM) comparison
ax2.plot(
    tmatrix_wavelengths,
    tmatrix_R_p,
    "g-",
    linewidth=2,
    label="TMatrix (P-polarization)",
    alpha=0.8,
)
ax2.plot(
    comsol_tm_wavelengths,
    comsol_tm_reflectance,
    "m--",
    linewidth=2,
    label="COMSOL (TM)",
    alpha=0.8,
    markersize=4,
)
ax2.set_xlabel("Wavelength (nm)", fontsize=12)
ax2.set_ylabel("Reflectance", fontsize=12)
ax2.set_title(
    "P-polarization (TM) Comparison @ 0°: TMatrix vs COMSOL",
    fontsize=13,
    fontweight="bold",
)
ax2.legend(fontsize=11, loc="best")
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot to plots folder
plots_dir = Path(__file__).parent / "plots"
plots_dir.mkdir(exist_ok=True)  # Create plots directory if it doesn't exist
output_file = plots_dir / "tmatrix_comsol_comparison_0DEG.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Comparison plot saved to: {output_file}")

# Print some statistics
print("\n" + "=" * 80)
print("Data Statistics:")
print("=" * 80)
print(f"TMatrix data points: {len(tmatrix_wavelengths)}")
print(
    f"  Wavelength range: {tmatrix_wavelengths.min():.2f} - {tmatrix_wavelengths.max():.2f} nm"
)
print(f"  R_s range: {tmatrix_R_s.min():.6f} - {tmatrix_R_s.max():.6f}")
print(f"  R_p range: {tmatrix_R_p.min():.6f} - {tmatrix_R_p.max():.6f}")

print(f"\nCOMSOL TE (S-polarization) data points: {len(comsol_te_wavelengths)}")
print(
    f"  Wavelength range: {comsol_te_wavelengths.min():.2f} - {comsol_te_wavelengths.max():.2f} nm"
)
print(
    f"  Reflectance range: {comsol_te_reflectance.min():.6f} - {comsol_te_reflectance.max():.6f}"
)

print(f"\nCOMSOL TM (P-polarization) data points: {len(comsol_tm_wavelengths)}")
print(
    f"  Wavelength range: {comsol_tm_wavelengths.min():.2f} - {comsol_tm_wavelengths.max():.2f} nm"
)
print(
    f"  Reflectance range: {comsol_tm_reflectance.min():.6f} - {comsol_tm_reflectance.max():.6f}"
)

# Interpolate to compare at common wavelengths - S-polarization (TE)
if len(comsol_te_wavelengths) > 0 and len(tmatrix_wavelengths) > 0:
    # Find overlapping wavelength range
    wl_min_s = max(tmatrix_wavelengths.min(), comsol_te_wavelengths.min())
    wl_max_s = min(tmatrix_wavelengths.max(), comsol_te_wavelengths.max())

    if wl_min_s < wl_max_s:
        # Create common wavelength grid
        common_wl_s = np.linspace(wl_min_s, wl_max_s, 200)

        # Interpolate both datasets to common grid
        tmatrix_R_s_interp = np.interp(common_wl_s, tmatrix_wavelengths, tmatrix_R_s)
        comsol_te_interp = np.interp(
            common_wl_s, comsol_te_wavelengths, comsol_te_reflectance
        )

        # Calculate differences
        diff_s = np.abs(tmatrix_R_s_interp - comsol_te_interp)
        max_diff_s = np.max(diff_s)
        mean_diff_s = np.mean(diff_s)
        rms_diff_s = np.sqrt(np.mean(diff_s**2))

        print(f"\n" + "=" * 80)
        print("S-polarization (TE) Comparison:")
        print("=" * 80)
        print(f"  Overlapping range: {wl_min_s:.2f} - {wl_max_s:.2f} nm")
        print(f"  Max difference: {max_diff_s:.6f}")
        print(f"  Mean difference: {mean_diff_s:.6f}")
        print(f"  RMS difference: {rms_diff_s:.6f}")

# Interpolate to compare at common wavelengths - P-polarization (TM)
if len(comsol_tm_wavelengths) > 0 and len(tmatrix_wavelengths) > 0:
    # Find overlapping wavelength range
    wl_min_p = max(tmatrix_wavelengths.min(), comsol_tm_wavelengths.min())
    wl_max_p = min(tmatrix_wavelengths.max(), comsol_tm_wavelengths.max())

    if wl_min_p < wl_max_p:
        # Create common wavelength grid
        common_wl_p = np.linspace(wl_min_p, wl_max_p, 200)

        # Interpolate both datasets to common grid
        tmatrix_R_p_interp = np.interp(common_wl_p, tmatrix_wavelengths, tmatrix_R_p)
        comsol_tm_interp = np.interp(
            common_wl_p, comsol_tm_wavelengths, comsol_tm_reflectance
        )

        # Calculate differences
        diff_p = np.abs(tmatrix_R_p_interp - comsol_tm_interp)
        max_diff_p = np.max(diff_p)
        mean_diff_p = np.mean(diff_p)
        rms_diff_p = np.sqrt(np.mean(diff_p**2))

        print(f"\n" + "=" * 80)
        print("P-polarization (TM) Comparison:")
        print("=" * 80)
        print(f"  Overlapping range: {wl_min_p:.2f} - {wl_max_p:.2f} nm")
        print(f"  Max difference: {max_diff_p:.6f}")
        print(f"  Mean difference: {mean_diff_p:.6f}")
        print(f"  RMS difference: {rms_diff_p:.6f}")

plt.show()
