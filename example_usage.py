"""
Example usage of the improved TMatrix implementation.

This script demonstrates key features and improvements in the new version.
"""

print("=" * 70)
print("Transfer Matrix Method - Example Usage")
print("=" * 70)
print()

# Import required modules
try:
    import numpy as np
    from TMatrix import TMatrix, RsTsRpTp

    HAS_NUMPY = True
except ImportError:
    print("Warning: numpy not available. Some examples will be skipped.")
    HAS_NUMPY = False

if not HAS_NUMPY:
    print("\nPlease install numpy to run examples:")
    print("  pip install numpy matplotlib")
    exit(1)

# Example 1: Basic single interface
print("Example 1: Air-Glass Interface")
print("-" * 70)
lda = 633e-9  # Red laser, 633 nm
theta = 0  # Normal incidence
d = [0, 0]  # No intermediate layers
eps = [1.0, 2.25]  # Air (n=1) and glass (n=1.5)

tm_s = TMatrix(lda, theta, "s", d, eps)
rt_s = RsTsRpTp(tm_s)

print(f"Wavelength: {lda * 1e9:.1f} nm")
print(f"Incident angle: {theta}°")
print(f"Materials: Air (n={np.sqrt(eps[0]):.1f}) → Glass (n={np.sqrt(eps[1]):.1f})")
print(f"\nResults (s-polarization):")
print(f"  Reflectance: {rt_s.R():.6f}")
print(f"  Transmittance: {rt_s.T():.6f}")
print(f"  R + T = {rt_s.R() + rt_s.T():.10f} (should be 1.0)")
print(f"  Energy conserved: {rt_s.energy_conservation_check()}")
print()

# Example 2: Quarter-wave anti-reflection coating
print("Example 2: Anti-Reflection Coating")
print("-" * 70)
lda_design = 550e-9  # Green light
n_substrate = 1.5
n_coating = np.sqrt(n_substrate)  # Optimal index
d_coating = lda_design / (4 * n_coating)

d = [0, d_coating, 0]
eps = [1.0, n_coating**2, n_substrate**2]

tm = TMatrix(lda_design, 0, "s", d, eps)
rt = RsTsRpTp(tm)

print(f"Design wavelength: {lda_design * 1e9:.1f} nm")
print(f"Coating: n={n_coating:.3f}, thickness={d_coating * 1e9:.1f} nm")
print(f"Substrate: n={n_substrate:.1f}")
print(f"\nReflectance at design wavelength: {rt.R():.6f}")
print(f"(Should be very low for good AR coating)")
print()

# Example 3: Brewster's Angle
print("Example 3: Brewster's Angle")
print("-" * 70)
n1, n2 = 1.0, 1.5
theta_brewster = np.arctan(n2 / n1) * 180 / np.pi

d = [0, 0]
eps = [n1**2, n2**2]

# Calculate at Brewster's angle
tm_p = TMatrix(633e-9, theta_brewster, "p", d, eps)
rt_p = RsTsRpTp(tm_p)

print(f"Interface: n₁={n1:.1f} → n₂={n2:.1f}")
print(f"Brewster's angle: {theta_brewster:.2f}°")
print(f"\nReflectance (p-pol) at Brewster's angle: {rt_p.R():.10f}")
print(f"(Should be essentially zero)")
print()

# Example 4: Angle Sweep with New Vectorization
print("Example 4: Angle Sweep (Vectorized)")
print("-" * 70)
thetas = np.linspace(0, 80, 81)
d = [0, 100e-9, 0]
eps = [1, 2.25, 1]

tm_s = TMatrix(633e-9, thetas, "s", d, eps)
tm_p = TMatrix(633e-9, thetas, "p", d, eps)

rt_s = RsTsRpTp(tm_s)
rt_p = RsTsRpTp(tm_p)

Rs = rt_s.R()
Rp = rt_p.R()

print(f"Number of angles: {len(thetas)}")
print(f"Angle range: {thetas[0]:.0f}° to {thetas[-1]:.0f}°")
print(f"\nSample results:")
print(f"  θ = 0°:  Rs = {Rs[0]:.4f}, Rp = {Rp[0]:.4f}")
print(f"  θ = 45°: Rs = {Rs[45]:.4f}, Rp = {Rp[45]:.4f}")
print(f"  θ = 80°: Rs = {Rs[80]:.4f}, Rp = {Rp[80]:.4f}")
print()

# Example 5: Lossy Material (Metal)
print("Example 5: Metal Film (Lossy Material)")
print("-" * 70)
eps_metal = -25 + 1.5j  # Complex permittivity
d_metal = 50e-9

d = [0, d_metal, 0]
eps = [1, eps_metal, 1]

tm = TMatrix(800e-9, 0, "p", d, eps)
rt = RsTsRpTp(tm)

R = rt.R()
T = rt.T()
A = 1 - R - T  # Absorption

print(f"Metal film: ε = {eps_metal}")
print(f"Thickness: {d_metal * 1e9:.1f} nm")
print(f"\nResults:")
print(f"  Reflectance: {R:.4f}")
print(f"  Transmittance: {T:.4f}")
print(f"  Absorption: {A:.4f}")
print(f"  R + T + A = {R + T + A:.6f}")
print()

# Example 6: Bragg Grating
print("Example 6: Bragg Grating (10 periods)")
print("-" * 70)
lda0 = 1.55e-6
n1, n2 = 2.0, 3.0
N_periods = 10

d = [0]
eps = [1]
for i in range(N_periods):
    d.extend([lda0 / (4 * n1), lda0 / (4 * n2)])
    eps.extend([n1**2, n2**2])
d.append(0)
eps.append(1)

tm = TMatrix(lda0, 0, "s", d, eps)
rt = RsTsRpTp(tm)

print(f"Design wavelength: {lda0 * 1e6:.2f} μm")
print(f"Layers: {len(eps)} total ({N_periods} periods)")
print(f"Refractive indices: n₁={n1:.1f}, n₂={n2:.1f}")
print(f"\nResults at design wavelength:")
print(f"  Reflectance: {rt.R():.6f}")
print(f"  Transmittance: {rt.T():.6f}")
print(f"  R + T = {rt.R() + rt.T():.10f}")
print()

# Example 7: Error Handling Demonstration
print("Example 7: Input Validation")
print("-" * 70)

# Test 1: Mismatched arrays
try:
    tm = TMatrix(1e-6, 0, "s", [0, 100e-9], [1, 2.25, 1])
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Caught expected error: {e}")

# Test 2: Invalid polarization
try:
    tm = TMatrix(1e-6, 0, "x", [0, 0], [1, 1])
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Caught expected error: {e}")

# Test 3: Negative wavelength
try:
    tm = TMatrix(-1e-6, 0, "s", [0, 0], [1, 1])
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Caught expected error: {e}")

print()

# Summary
print("=" * 70)
print("Examples Complete!")
print("=" * 70)
print("\nKey Improvements Demonstrated:")
print("  ✓ Comprehensive error handling with informative messages")
print("  ✓ Type hints and detailed docstrings")
print("  ✓ Energy conservation checking")
print("  ✓ Improved numerical stability")
print("  ✓ Support for complex permittivities")
print("  ✓ Efficient vectorized angle sweeps")
print("\nFor more details, see README.md")
print()
