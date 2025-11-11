# Transfer Matrix Method for Multi-Layer Optical Structures

A comprehensive Python implementation of the Transfer Matrix Method (TMM) for calculating optical properties (reflectance and transmittance) of multi-layer photonic structures. Supports both s-polarization (TE) and p-polarization (TM) at arbitrary incident angles across multiple wavelengths.

## Table of Contents

- [Features](#features)
- [Layered Model](#layered-model)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Physical Background](#physical-background)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Energy Conservation](#energy-conservation)
- [References](#references)

## Features

✨ **Comprehensive Functionality**

- Calculate reflectance and transmittance for multi-layer structures
- Support for 0 to N layers (including single interface)
- Both s-polarization (TE) and p-polarization (TM)
- Arbitrary incident angles (0° to 90°)
- Wavelength array support for efficient sweeps
- Handle complex permittivities (lossy materials)
- Proper power flow correction for different incident/exit media

🚀 **Performance & Stability**

- Vectorized operations for efficient wavelength sweeps
- Numerical stability using cos(θ) instead of kz/k0 ratios
- Comprehensive input validation
- Energy conservation verified (R + T = 1.0 for lossless materials)

📚 **Well Documented**

- Comprehensive docstrings following NumPy conventions
- Type hints for better IDE support
- Multiple test examples
- Clear error messages

## Layered Model

The multi-layer structure is defined as a stack of layers between two semi-infinite media:

```
┌─────────────────────────────────────────────────────────────┐
│                    Incident Medium (ε₁)                      │
│                    (e.g., Air, n = 1.0)                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Layer 1 (ε₂, d₁)                            │   │
│  │         (e.g., SiO₂, n = 1.45)                      │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │         Layer 2 (ε₃, d₂)                            │   │
│  │         (e.g., Si, n = 3.5)                         │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │         Layer 3 (ε₄, d₃)                            │   │
│  │         (e.g., SiO₂, n = 1.45)                      │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │         ...                                         │   │
│  │         (more layers)                                │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │         Layer N (εₙ₊₁, dₙ)                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                    Exit Medium (εₙ₊₂)                        │
│                    (e.g., Glass, n = 1.5)                    │
└─────────────────────────────────────────────────────────────┘

Incident Wave (θ) →  →  →  →  →  →  →  →  →  →  →  →  →  →  →  Transmitted Wave
                    ↓
                Reflected Wave
```

### Structure Definition

The structure is mathematically represented as:

**Incident Medium (ε₁)** | **Layer 1 (ε₂, d₁)** | **Layer 2 (ε₃, d₂)** | ... | **Layer N (εₙ₊₁, dₙ)** | **Exit Medium (εₙ₊₂)**

Where:
- **εᵢ**: Permittivity of medium/layer i (ε = n² for non-magnetic materials)
- **dᵢ**: Thickness of layer i (in meters)
- **θ**: Incident angle (in degrees)
- **λ**: Wavelength (in meters)

### Special Cases

1. **Zero Layers (Single Interface)**: 
   ```
   Incident Medium (ε₁) | Exit Medium (ε₂)
   ```
   This is a simple Fresnel interface.

2. **One Layer**:
   ```
   Incident Medium (ε₁) | Layer (ε₂, d) | Exit Medium (ε₃)
   ```

3. **Multiple Layers**:
   ```
   Incident Medium | Layer 1 | Layer 2 | ... | Layer N | Exit Medium
   ```

## Installation

### Requirements

```bash
numpy >= 1.20.0
matplotlib >= 3.3.0  # for plotting examples
```

### Setup

Clone this repository or download the files:

```bash
git clone <repository-url>
cd TransferMatrix
```

Or simply copy `TMatrix.py` to your project directory.

## Quick Start

### Basic Example: Single Interface (0 layers)

```python
import numpy as np
from TMatrix import MultiLayerStructure

# Define wavelength array
wavelengths = np.array([500e-9, 600e-9, 700e-9])  # 500, 600, 700 nm
angle_degrees = 30.0  # 30 degrees incident angle

# Create structure: Air | Glass (single interface)
ml = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=angle_degrees,
    polarization='s',  # s-polarization (TE)
    layers=[],  # Empty list = single interface
    eps_incident=1.0,  # Air
    eps_exit=2.25,  # Glass (n=1.5, so n²=2.25)
)

# Calculate reflectance and transmittance
R = ml.reflectance()  # Shape: (3,)
T = ml.transmittance()  # Shape: (3,)

print(f"Reflectance: {R}")
print(f"Transmittance: {T}")
print(f"R + T = {R + T}")  # Should be ~1.0 for lossless materials
```

### Basic Example: Single Layer

```python
import numpy as np
from TMatrix import Layer, MultiLayerStructure

# Define wavelength array
wavelengths = np.linspace(400e-9, 800e-9, 100)  # 400-800 nm, 100 points
angle_degrees = 45.0  # 45 degrees

# Create a single layer: Air | SiO₂ (100 nm) | Glass
layer = Layer(
    thickness=100e-9,  # 100 nm
    optical_property={"type": "permittivity", "value": 2.13}  # SiO₂ (n≈1.46)
)

ml = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=angle_degrees,
    polarization='p',  # p-polarization (TM)
    layers=[layer],
    eps_incident=1.0,  # Air
    eps_exit=2.25,  # Glass
)

# Calculate optical properties
R = ml.reflectance()
T = ml.transmittance()

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(wavelengths * 1e9, R, label='Reflectance')
plt.plot(wavelengths * 1e9, T, label='Transmittance')
plt.plot(wavelengths * 1e9, R + T, '--', label='R + T')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance / Transmittance')
plt.title('Single Layer: Air | SiO₂ (100 nm) | Glass')
plt.legend()
plt.grid(True)
plt.show()
```

### Basic Example: Multi-Layer Structure

```python
import numpy as np
from TMatrix import Layer, MultiLayerStructure

# Create a Bragg grating: alternating high/low index layers
wavelengths = np.linspace(500e-9, 700e-9, 200)
angle_degrees = 0.0  # Normal incidence

layers = []
for i in range(5):  # 5 periods
    # High index layer (Si, n=3.5)
    layers.append(Layer(
        thickness=50e-9,
        optical_property={"type": "refractive_index", "value": 3.5}
    ))
    # Low index layer (SiO₂, n=1.46)
    layers.append(Layer(
        thickness=120e-9,
        optical_property={"type": "refractive_index", "value": 1.46}
    ))

ml = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=angle_degrees,
    polarization='s',
    layers=layers,
    eps_incident=1.0,  # Air
    eps_exit=1.0,  # Air
)

R = ml.reflectance()
T = ml.transmittance()

# Energy conservation check
print(f"Max |R + T - 1|: {np.max(np.abs(R + T - 1.0)):.2e}")
```

## Physical Background

### Transfer Matrix Method

The Transfer Matrix Method relates electromagnetic field amplitudes across interfaces and through layers. For a multi-layer structure, the total transfer matrix is constructed by multiplying interface and propagation matrices:

**T_total = F_out_inv @ F_N @ P_N @ F_N_inv @ ... @ F_1 @ P_1 @ F_1_inv @ F_in**

Where:
- **F_in**: Forward matrix for incident medium
- **F_i_inv**: Inverse matrix for layer i (interface from previous medium to layer i)
- **P_i**: Propagation matrix through layer i
- **F_i**: Forward matrix for layer i (interface from layer i to next medium)
- **F_out_inv**: Inverse matrix for exit medium

### Matrix Structure

**Interface Matrix (F)**:
- **s-polarization**: F = [[1, 1], [-a, a]] where a = √ε·cos(θ) = kz/k₀
- **p-polarization**: F = [[b, -b], [1, 1]] where b = cos(θ)/√ε = kz/(k₀·ε)

**Propagation Matrix (P)**:
```
P = [[exp(i·kz·d), 0], [0, exp(-i·kz·d)]]
```

### Reflectance and Transmittance

**Reflectance**:
```
R = |r|², where r = -T₂₁ / T₂₂
```

**Transmittance**:
- **Zero layers (single interface)**: Full power flow correction applied
- **One or more layers**: Simplified formula (power flow handled by matrix construction)

### Wave Vector Components

- **k₀ = 2π/λ**: Free-space wave number
- **kx = k₀·√ε·sin(θ)**: Tangential component (conserved across interfaces via Snell's law)
- **kz = √(k₀²·ε - kx²)**: Normal component (varies with permittivity)

### Polarization Modes

- **s-polarization (TE)**: Electric field perpendicular to plane of incidence
- **p-polarization (TM)**: Magnetic field perpendicular to plane of incidence

## API Reference

### MultiLayerStructure Class

```python
MultiLayerStructure(
    wavelengths: float | np.ndarray,
    angle_degrees: float,
    polarization: str,
    layers: list[Layer],
    eps_incident: float | complex | np.ndarray,
    eps_exit: float | complex | np.ndarray,
)
```

**Parameters:**

- `wavelengths` (float or array): Wavelength(s) in meters
- `angle_degrees` (float): Incident angle in degrees, range [0, 90)
- `polarization` (str): 's' for TE, 'p' for TM
- `layers` (list[Layer]): List of Layer objects, ordered from incident to exit
- `eps_incident` (float or array): Permittivity of incident medium
- `eps_exit` (float or array): Permittivity of exit medium

**Methods:**

- `total_transfer_matrix()`: Calculate total transfer matrix, shape (n_wavelengths, 2, 2)
- `reflectance()`: Calculate reflectance R = |r|², shape (n_wavelengths,)
- `transmittance()`: Calculate transmittance with power flow correction, shape (n_wavelengths,)
- `wave_vectors()`: Get all wave vector components (k0, kx, kz_incident, kz_exit, kz_layers)

### Layer Class

```python
Layer(
    thickness: float,
    optical_property: dict,
)
```

**Parameters:**

- `thickness` (float): Layer thickness in meters (must be >= 0)
- `optical_property` (dict): Dictionary with keys:
  - `'type'`: Either `'permittivity'` or `'refractive_index'`
  - `'value'`: The permittivity or refractive index value

**Example:**

```python
# Using permittivity
layer1 = Layer(
    thickness=100e-9,
    optical_property={"type": "permittivity", "value": 2.25}
)

# Using refractive index (automatically converted to permittivity: ε = n²)
layer2 = Layer(
    thickness=200e-9,
    optical_property={"type": "refractive_index", "value": 1.5}
)
```

## Examples

### Example 1: Air-Silicon-Silicon Dioxide Structure

```python
import numpy as np
from TMatrix import Layer, MultiLayerStructure

wavelengths = np.linspace(400e-9, 1000e-9, 300)
angle_degrees = 0.0  # Normal incidence

# Structure: Air | Si (200 nm) | SiO₂ (500 nm) | Air
layers = [
    Layer(
        thickness=200e-9,
        optical_property={"type": "refractive_index", "value": 3.5}  # Si
    ),
    Layer(
        thickness=500e-9,
        optical_property={"type": "refractive_index", "value": 1.46}  # SiO₂
    ),
]

ml = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=angle_degrees,
    polarization='s',
    layers=layers,
    eps_incident=1.0,  # Air
    eps_exit=1.0,  # Air
)

R = ml.reflectance()
T = ml.transmittance()

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(wavelengths * 1e9, R, label='R')
plt.plot(wavelengths * 1e9, T, label='T')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance / Transmittance')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(wavelengths * 1e9, R + T, label='R + T')
plt.axhline(y=1.0, color='r', linestyle='--', label='Energy Conservation')
plt.xlabel('Wavelength (nm)')
plt.ylabel('R + T')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Example 2: Oblique Incidence with Different Media

```python
import numpy as np
from TMatrix import MultiLayerStructure

wavelengths = np.array([633e-9])  # He-Ne laser
angles = np.linspace(0, 85, 100)  # 0 to 85 degrees

R_s = []
R_p = []
T_s = []
T_p = []

for angle in angles:
    # s-polarization
    ml_s = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=angle,
        polarization='s',
        layers=[],  # Single interface
        eps_incident=1.0,  # Air
        eps_exit=2.25,  # Glass
    )
    R_s.append(ml_s.reflectance()[0])
    T_s.append(ml_s.transmittance()[0])
    
    # p-polarization
    ml_p = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=angle,
        polarization='p',
        layers=[],
        eps_incident=1.0,
        eps_exit=2.25,
    )
    R_p.append(ml_p.reflectance()[0])
    T_p.append(ml_p.transmittance()[0])

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(angles, R_s, label='R_s (TE)')
plt.plot(angles, R_p, label='R_p (TM)')
plt.plot(angles, T_s, '--', label='T_s (TE)')
plt.plot(angles, T_p, '--', label='T_p (TM)')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Reflectance / Transmittance')
plt.title('Air-Glass Interface: Reflectance vs Angle')
plt.legend()
plt.grid(True)
plt.show()
```

### Example 3: Wavelength-Dependent Permittivity

```python
import numpy as np
from TMatrix import Layer, MultiLayerStructure

wavelengths = np.linspace(400e-9, 800e-9, 200)

# Wavelength-dependent permittivity (simple dispersion model)
def silicon_permittivity(lda):
    """Simple dispersion model for silicon"""
    n = 3.5 + 0.1 * (lda - 600e-9) / 100e-9
    return n**2

eps_si = silicon_permittivity(wavelengths)

# Create layer with wavelength-dependent permittivity
layer = Layer(
    thickness=100e-9,
    optical_property={"type": "permittivity", "value": eps_si}
)

ml = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=0.0,
    polarization='s',
    layers=[layer],
    eps_incident=1.0,
    eps_exit=1.0,
)

R = ml.reflectance()
T = ml.transmittance()

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(wavelengths * 1e9, R, label='Reflectance')
plt.plot(wavelengths * 1e9, T, label='Transmittance')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance / Transmittance')
plt.title('Si Layer with Wavelength-Dependent Permittivity')
plt.legend()
plt.grid(True)
plt.show()
```

## Energy Conservation

The implementation ensures energy conservation (R + T = 1.0) for lossless materials. This is verified through comprehensive tests:

- ✅ Normal incidence with 0-10 layers
- ✅ Oblique incidence with 0-10 layers
- ✅ Both s and p polarizations
- ✅ Same and different incident/exit media
- ✅ Multiple wavelengths

To verify energy conservation in your calculations:

```python
R = ml.reflectance()
T = ml.transmittance()
R_plus_T = R + T

# Check energy conservation
max_deviation = np.max(np.abs(R_plus_T - 1.0))
print(f"Max |R + T - 1|: {max_deviation:.2e}")

# Should be very close to zero for lossless materials
assert np.allclose(R_plus_T, 1.0, atol=1e-10)
```

## Performance

### Optimization Tips

1. **Vectorize wavelength sweeps**: Pass arrays instead of looping
   ```python
   # Good: Single call with array
   wavelengths = np.linspace(400e-9, 800e-9, 1000)
   ml = MultiLayerStructure(wavelengths, angle, pol, layers, eps_in, eps_out)
   
   # Less efficient: Loop
   for lda in wavelengths:
       ml = MultiLayerStructure(lda, angle, pol, layers, eps_in, eps_out)
   ```

2. **Use broadcasting**: Scalar inputs are automatically broadcast
   ```python
   # eps_incident is scalar, automatically broadcast to match wavelengths
   ml = MultiLayerStructure(wavelengths, angle, pol, layers, 1.0, eps_out)
   ```

### Benchmarks

Typical performance on modern hardware:
- Single calculation: ~0.01 ms
- 1000 wavelengths: ~1-2 ms (vectorized)
- Complex permittivities: ~2x slower than real

## Limitations

1. **Planar Interfaces**: Assumes infinite planar interfaces
2. **Linear Media**: No nonlinear optical effects
3. **Numerical Precision**: Very large or very small wavelengths may cause issues
4. **Grazing Angles**: Accuracy may degrade near 90° incidence
5. **Lossy Materials**: Complex permittivities supported, but energy conservation only holds for lossless materials

## Best Practices

1. **Use appropriate units**: All lengths in meters, angles in degrees
2. **Handle complex permittivities**: Use `eps = n_real**2 + 1j*n_imag` format
3. **Validate inputs**: The code includes comprehensive validation
4. **Start simple**: Test with known cases (Fresnel equations, etc.)
5. **Check energy conservation**: Verify R + T = 1.0 for lossless materials
6. **Use refractive_index for clarity**: More intuitive than permittivity

## Error Handling

The implementation includes comprehensive error checking:

```python
# Raises ValueError if inputs are invalid
try:
    ml = MultiLayerStructure(
        wavelengths=-1e-6,  # Negative wavelength
        angle_degrees=0,
        polarization='s',
        layers=[],
        eps_incident=1.0,
        eps_exit=2.25,
    )
except ValueError as e:
    print(f"Error: {e}")

# Warns if angles are out of physical range
ml = MultiLayerStructure(
    wavelengths=1e-6,
    angle_degrees=95,  # Warning issued
    polarization='s',
    layers=[],
    eps_incident=1.0,
    eps_exit=2.25,
)

# Raises error for exactly 90° (grazing incidence)
try:
    ml = MultiLayerStructure(
        wavelengths=1e-6,
        angle_degrees=90,  # Error
        polarization='s',
        layers=[],
        eps_incident=1.0,
        eps_exit=2.25,
    )
except ValueError as e:
    print(f"Error: {e}")
```

## References

1. Born, M., & Wolf, E. (1999). *Principles of Optics* (7th ed.). Cambridge University Press.
2. Yeh, P. (2005). *Optical Waves in Layered Media*. Wiley-Interscience.
3. Saleh, B. E. A., & Teich, M. C. (2007). *Fundamentals of Photonics* (2nd ed.). Wiley.
4. Macleod, H. A. (2010). *Thin-Film Optical Filters* (4th ed.). CRC Press.

## License

This code is provided for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure:

- Code follows PEP 8 style guidelines
- All functions have comprehensive docstrings
- Type hints are included
- Tests pass (run `test_*.py` files)

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Version**: 2.0  
**Last Updated**: 2025  
**Python Compatibility**: 3.7+
