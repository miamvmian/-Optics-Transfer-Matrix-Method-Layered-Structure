#!/usr/bin/env python
"""
Calculate reflectance and transmittance for air|Si|SiO2 structure.

Structure:
- Incident medium: Air (ε = 1.0)
- Layer 1: Silicon (340 nm thick, wavelength-dependent permittivity)
- Exit medium: SiO2 (ε = 2.10, n ≈ 1.45)
"""

import numpy as np
import matplotlib.pyplot as plt
from TMatrix import Layer, MultiLayerStructure
from Si_refractive_fitting.silicon_ri import SiliconRefractiveIndex

print("=" * 80)
print("Air|Si|SiO2 Structure - Reflectance and Transmittance Calculation")
print("=" * 80)

# Test parameters
wavelengths_nm = np.linspace(600, 700, 11)  # Wavelengths in nanometers
wavelengths = wavelengths_nm * 1e-9  # Convert to meters for TMatrix
angle_degrees = 0.0  # Normal incidence
si_thickness = 340e-9  # Silicon layer thickness: 340 nm

# SiO2 properties (exit medium)
eps_SiO2 = 2.10  # Permittivity of SiO2 (n ≈ 1.45)

# Create Silicon refractive index evaluator
si_ri = SiliconRefractiveIndex(fit_method="polynomial")

# Get Silicon permittivity at each wavelength
# Note: TMatrix expects Im(ε) < 0 for lossy materials (opposite of standard convention)
# Standard: ε = (n + iκ)² gives Im(ε) > 0, but TMatrix needs Im(ε) < 0
# Solution: Use conjugated epsilon
epsilon_Si = np.conj(
    si_ri.epsilon(wavelengths_nm)
)  # Complex permittivity (conjugated for TMatrix)
n_Si = si_ri.n(wavelengths_nm)  # Real part of refractive index
kappa_Si = si_ri.kappa(wavelengths_nm)  # Imaginary part (extinction coefficient)

print(f"\nStructure Configuration:")
print(f"  Incident medium: Air (ε = 1.0)")
print(f"  Layer: Silicon, thickness = {si_thickness*1e9:.0f} nm")
print(f"  Exit medium: SiO2 (ε = {eps_SiO2:.2f})")
print(f"  Wavelength range: {wavelengths_nm.min():.1f} - {wavelengths_nm.max():.1f} nm")
print(f"  Incident angle: {angle_degrees}° (normal incidence)")

print(f"\nSilicon Properties:")
print(f"  {'Wavelength (nm)':<15} {'n':<10} {'κ':<12} {'ε_real':<12} {'ε_imag':<12}")
print("-" * 70)
for i, wl in enumerate(wavelengths_nm):
    print(
        f"  {wl:<15.1f} {n_Si[i]:<10.4f} {kappa_Si[i]:<12.6f} "
        f"{np.real(epsilon_Si[i]):<12.4f} {np.imag(epsilon_Si[i]):<12.4f}"
    )

# Note: Layer class only accepts scalar permittivity values
# We calculate wavelength-by-wavelength to handle wavelength-dependent Silicon permittivity

# Calculate for both polarizations, wavelength by wavelength
results = {}

for polarization in ["s", "p"]:
    print(f"\n{'='*80}")
    print(
        f"Calculating for {polarization.upper()}-polarization ({'TE' if polarization == 's' else 'TM'})"
    )
    print(f"{'='*80}")

    R_list = []
    T_list = []

    # Calculate for each wavelength separately
    for i, (wl, wl_nm, eps_si) in enumerate(
        zip(wavelengths, wavelengths_nm, epsilon_Si)
    ):
        # Create layer with wavelength-specific permittivity
        layer = Layer(
            thickness=si_thickness,
            optical_property={"type": "permittivity", "value": eps_si},
        )

        # Create multi-layer structure for this wavelength
        ml = MultiLayerStructure(
            wavelengths=np.array([wl]),  # Single wavelength
            angle_degrees=angle_degrees,
            polarization=polarization,
            layers=[layer],
            eps_incident=1.0,  # Air
            eps_exit=eps_SiO2,  # SiO2
        )

        # Calculate reflectance and transmittance
        R = ml.reflectance()[0]  # Get scalar value
        T = ml.transmittance()[0]  # Get scalar value

        R_list.append(R)
        T_list.append(T)

    # Convert to arrays
    R = np.array(R_list)
    T = np.array(T_list)

    # Store results
    results[polarization] = {"R": R, "T": T, "R_plus_T": R + T}

    # Display results
    print(
        f"\n{'Wavelength (nm)':<15} {'Reflectance (R)':<18} {'Transmittance (T)':<18} {'R + T':<12}"
    )
    print("-" * 70)
    for i, wl in enumerate(wavelengths_nm):
        print(f"  {wl:<15.1f} {R[i]:<18.6f} {T[i]:<18.6f} {R[i] + T[i]:<12.6f}")

    print(f"\nSummary:")
    print(f"  Average R: {np.mean(R):.6f}")
    print(f"  Average T: {np.mean(T):.6f}")
    print(f"  Average R + T: {np.mean(R + T):.6f}")
    print(f"  R range: [{np.min(R):.6f}, {np.max(R):.6f}]")
    print(f"  T range: [{np.min(T):.6f}, {np.max(T):.6f}]")

# Plot results
print(f"\n{'='*80}")
print("Generating plots...")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Reflectance vs Wavelength
ax = axes[0, 0]
ax.plot(
    wavelengths_nm,
    results["s"]["R"],
    "o-",
    label="s-polarization (TE)",
    linewidth=2,
    markersize=6,
)
ax.plot(
    wavelengths_nm,
    results["p"]["R"],
    "s-",
    label="p-polarization (TM)",
    linewidth=2,
    markersize=6,
)
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("Reflectance (R)", fontsize=12)
ax.set_title("Reflectance vs Wavelength", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(wavelengths_nm.min(), wavelengths_nm.max())

# Plot 2: Transmittance vs Wavelength
ax = axes[0, 1]
ax.plot(
    wavelengths_nm,
    results["s"]["T"],
    "o-",
    label="s-polarization (TE)",
    linewidth=2,
    markersize=6,
)
ax.plot(
    wavelengths_nm,
    results["p"]["T"],
    "s-",
    label="p-polarization (TM)",
    linewidth=2,
    markersize=6,
)
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("Transmittance (T)", fontsize=12)
ax.set_title("Transmittance vs Wavelength", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(wavelengths_nm.min(), wavelengths_nm.max())

# Plot 3: R + T (Energy Conservation Check)
ax = axes[1, 0]
ax.plot(
    wavelengths_nm,
    results["s"]["R_plus_T"],
    "o-",
    label="s-polarization (TE)",
    linewidth=2,
    markersize=6,
)
ax.plot(
    wavelengths_nm,
    results["p"]["R_plus_T"],
    "s-",
    label="p-polarization (TM)",
    linewidth=2,
    markersize=6,
)
ax.axhline(
    y=1.0, color="red", linestyle="--", linewidth=2, label="R + T = 1 (lossless)"
)
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("R + T", fontsize=12)
ax.set_title("Energy Conservation (R + T)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(wavelengths_nm.min(), wavelengths_nm.max())
# Note: R + T < 1 because Silicon is lossy (has extinction coefficient)

# Plot 4: Absorption (1 - R - T)
ax = axes[1, 1]
absorption_s = 1 - results["s"]["R"] - results["s"]["T"]
absorption_p = 1 - results["p"]["R"] - results["p"]["T"]
ax.plot(
    wavelengths_nm,
    absorption_s,
    "o-",
    label="s-polarization (TE)",
    linewidth=2,
    markersize=6,
)
ax.plot(
    wavelengths_nm,
    absorption_p,
    "s-",
    label="p-polarization (TM)",
    linewidth=2,
    markersize=6,
)
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("Absorption (1 - R - T)", fontsize=12)
ax.set_title("Absorption in Silicon Layer", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(wavelengths_nm.min(), wavelengths_nm.max())

plt.tight_layout()
plt.savefig("air_Si_SiO2_reflectance_transmittance.png", dpi=300, bbox_inches="tight")
print("Plot saved to: air_Si_SiO2_reflectance_transmittance.png")
plt.show()

print(f"\n{'='*80}")
print("Calculation Complete!")
print(f"{'='*80}")
