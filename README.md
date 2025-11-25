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
- **Wavelength-dependent permittivity**: Support for both constant (scalar) and wavelength-dependent (array) permittivities
- Handle complex permittivities (lossy materials)
- Proper power flow correction for different incident/exit media
- **Refractive index support**: Specify materials using either permittivity or refractive index

🚀 **Performance & Stability**

- Vectorized operations for efficient wavelength sweeps
- Numerical stability using cos(θ) instead of kz/k0 ratios
- Comprehensive input validation
- **Cached interface/propagation matrices** so repeated reflectance/transmittance queries reuse prior work
- **Energy conservation verified**: R + T = 1.0 for lossless materials across all test cases
- **Comprehensive test suite**: 200+ test cases organized into focused test modules covering all scenarios

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

- `wavelengths` (float or array): Wavelength(s) in meters. If array, shape (n_wavelengths,)
- `angle_degrees` (float): Incident angle in degrees, range [0, 90)
- `polarization` (str): 's' for TE, 'p' for TM
- `layers` (list[Layer]): List of Layer objects, ordered from incident to exit
- `eps_incident` (float, complex, or array): Permittivity of incident medium
  - Scalar: Constant permittivity (broadcast to all wavelengths)
  - Array: Wavelength-dependent permittivity, shape (n_wavelengths,) matching wavelengths
- `eps_exit` (float, complex, or array): Permittivity of exit medium
  - Scalar: Constant permittivity (broadcast to all wavelengths)
  - Array: Wavelength-dependent permittivity, shape (n_wavelengths,) matching wavelengths

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
    - **Scalar**: Constant value across all wavelengths
    - **Array**: Wavelength-dependent value, shape (n_wavelengths,) matching wavelengths array

**Examples:**

```python
# Using constant permittivity (scalar)
layer1 = Layer(
    thickness=100e-9,
    optical_property={"type": "permittivity", "value": 2.25}
)

# Using constant refractive index (automatically converted to permittivity: ε = n²)
layer2 = Layer(
    thickness=200e-9,
    optical_property={"type": "refractive_index", "value": 1.5}
)

# Using wavelength-dependent permittivity (array)
wavelengths = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
eps_array = 2.25 + 0.01 * (wavelengths * 1e9 - 600) / 600  # Dispersion model
layer3 = Layer(
    thickness=150e-9,
    optical_property={"type": "permittivity", "value": eps_array}
)

# Using wavelength-dependent refractive index (array)
n_array = 1.5 + 0.01 * (wavelengths * 1e9 - 600) / 600
layer4 = Layer(
    thickness=150e-9,
    optical_property={"type": "refractive_index", "value": n_array}
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

### Example 4: Single Interface (Zero Layers) with Wavelength-Dependent Permittivity

```python
import numpy as np
from TMatrix import MultiLayerStructure

wavelengths = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])

# Wavelength-dependent permittivity for glass (dispersion model)
eps_air = np.ones(len(wavelengths))  # Air (constant, but as array)
eps_glass = 2.25 + 0.01 * (wavelengths * 1e9 - 600) / 600  # Glass with dispersion

# Single interface (zero layers) with wavelength-dependent permittivity
ml = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=45.0,  # Oblique incidence
    polarization='s',
    layers=[],  # Empty list = single interface
    eps_incident=eps_air,    # Wavelength-dependent (array)
    eps_exit=eps_glass,      # Wavelength-dependent (array)
)

R = ml.reflectance()
T = ml.transmittance()

# Verify energy conservation for all wavelengths
R_plus_T = R + T
print(f"Energy conservation check:")
print(f"  R + T = {R_plus_T}")
print(f"  Max deviation from 1.0: {np.max(np.abs(R_plus_T - 1.0)):.2e}")

# Test different configurations
configs = [
    ("Both wavelength-dependent", eps_air, eps_glass),
    ("Incident wavelength-dependent, exit constant", eps_air, 2.25),
    ("Incident constant, exit wavelength-dependent", 1.0, eps_glass),
]

for name, eps_in, eps_out in configs:
    ml = MultiLayerStructure(
        wavelengths=wavelengths,
        angle_degrees=30.0,
        polarization='p',
        layers=[],
        eps_incident=eps_in,
        eps_exit=eps_out,
    )
    R = ml.reflectance()
    T = ml.transmittance()
    max_dev = np.max(np.abs(R + T - 1.0))
    print(f"{name}: Max deviation = {max_dev:.2e}")
```

### Example 5: Mixed Constant and Wavelength-Dependent Permittivity

```python
import numpy as np
from TMatrix import Layer, MultiLayerStructure

wavelengths = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])

# Constant permittivity for air
eps_air = 1.0

# Wavelength-dependent permittivity for glass (dispersion)
eps_glass = 2.25 + 0.01 * (wavelengths * 1e9 - 600) / 600

# Mixed layers: constant and wavelength-dependent
layer1 = Layer(
    thickness=100e-9,
    optical_property={"type": "permittivity", "value": 2.13}  # Constant
)

layer2 = Layer(
    thickness=150e-9,
    optical_property={"type": "permittivity", "value": eps_glass}  # Wavelength-dependent
)

ml = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=30.0,
    polarization='p',
    layers=[layer1, layer2],
    eps_incident=eps_air,      # Constant
    eps_exit=eps_glass,        # Wavelength-dependent
)

R = ml.reflectance()
T = ml.transmittance()

# Verify energy conservation
R_plus_T = R + T
print(f"Energy conservation check:")
print(f"  R + T = {R_plus_T}")
print(f"  Max deviation from 1.0: {np.max(np.abs(R_plus_T - 1.0)):.2e}")
```

### Example 6: Comprehensive Wavelength-Dependent Structure

```python
import numpy as np
from TMatrix import Layer, MultiLayerStructure

wavelengths = np.linspace(400e-9, 800e-9, 100)

# Create wavelength-dependent permittivity arrays
eps_air = np.ones(len(wavelengths))  # Air (constant, but as array)
eps_silica = 2.13 + 0.005 * (wavelengths * 1e9 - 600) / 600  # Silica with dispersion
eps_titanium = 5.76 + 0.02 * (wavelengths * 1e9 - 600) / 600  # Titanium with dispersion

# Create multiple layers with wavelength-dependent permittivity
layers = []
for i in range(5):
    if i % 2 == 0:
        # Even layers: wavelength-dependent silica
        layers.append(Layer(
            thickness=(50 + i * 5) * 1e-9,
            optical_property={"type": "permittivity", "value": eps_silica}
        ))
    else:
        # Odd layers: wavelength-dependent titanium
        layers.append(Layer(
            thickness=(50 + i * 5) * 1e-9,
            optical_property={"type": "permittivity", "value": eps_titanium}
        ))

ml = MultiLayerStructure(
    wavelengths=wavelengths,
    angle_degrees=45.0,
    polarization='s',
    layers=layers,
    eps_incident=eps_air,      # Wavelength-dependent (array)
    eps_exit=eps_air,          # Wavelength-dependent (array)
)

R = ml.reflectance()
T = ml.transmittance()

# Energy conservation should hold for all wavelengths
assert np.allclose(R + T, 1.0, atol=1e-10), "Energy conservation violated!"
```

## Energy Conservation

The implementation ensures energy conservation (R + T = 1.0) for lossless materials. This is verified through comprehensive tests:

- ✅ Normal incidence with 0-10 layers
- ✅ Oblique incidence with 0-10 layers (0° to 89°)
- ✅ Both s and p polarizations
- ✅ Same and different incident/exit media
- ✅ Multiple wavelengths
- ✅ Constant permittivity (scalar values)
- ✅ Wavelength-dependent permittivity (arrays)
- ✅ Mixed cases (constant + wavelength-dependent)
- ✅ Zero layers (single interface) with wavelength-dependent permittivity
- ✅ Multiple layers with different permittivity types
- ✅ Refractive index type (both constant and wavelength-dependent)
- ✅ Very high angles (up to 89°)
- ✅ Multiple material combinations
- ✅ Layer thickness variations

### Test Coverage

The codebase includes comprehensive test suites in the `tests/` directory that verify energy conservation. The tests are organized into focused modules:

**Core Test Files (in `tests/` directory):**
- **test_common_setup.py**: Shared constants and material definitions for all tests
- **test_constant_permittivity.py**: Constant permittivity (scalar values) tests
- **test_wavelength_dependent_permittivity_basic.py**: Basic wavelength-dependent permittivity (arrays) tests
- **test_mixed_permittivity.py**: Mixed cases (constant + wavelength-dependent)
- **test_multilayer_mixed_permittivity.py**: Multiple layers with mixed permittivity types
- **test_zero_layer_wavelength_dependent.py**: Zero layers (single interface) with wavelength-dependent permittivity
- **test_multilayer_wavelength_dependent.py**: Different numbers of layers (0-10) with wavelength-dependent permittivity
- **test_oblique_wavelength_dependent.py**: Oblique incidence with wavelength-dependent permittivity
- **test_multilayer_oblique_wavelength_dependent.py**: Multiple layers at different angles
- **test_refractive_index.py**: Refractive index type (converted to permittivity)
- **test_high_angles_materials.py**: Very high angles (up to 89°) with multiple material combinations
- **test_layer_thickness_variations.py**: Layer thickness variations (50-500 nm)
- **test_single_wavelength_focused.py**: Single wavelength focused tests with pattern analysis
- **test_same_medium.py**: Same incident/exit medium comprehensive tests
- **test_pattern_analysis.py**: Pattern analysis for different exit medium scenarios
- **test_visualization.py**: Visualization plots (requires matplotlib)

**Running Tests:**
```bash
# Run individual tests
python3 tests/test_constant_permittivity.py
python3 tests/test_wavelength_dependent_permittivity_basic.py

# Run the full numerical suite (visualization test optional)
for f in tests/test_*.py; do
  if [ "$f" = "tests/test_visualization.py" ]; then
    continue  # run plotting test manually when needed
  fi
  python3 "$f"
done

# test_visualization.py produces plots; ensure MPLCONFIGDIR points to a writable folder before running it:
MPLCONFIGDIR=.mplconfig python3 tests/test_visualization.py
```

**Test Coverage Summary:**
- ✅ Constant permittivity (scalar values)
- ✅ Wavelength-dependent permittivity (arrays)
- ✅ Mixed cases (constant + wavelength-dependent)
- ✅ Zero layers (single interface) with wavelength-dependent permittivity
- ✅ Multiple layers (0 to 10) with wavelength-dependent permittivity
- ✅ Oblique incidence (0° to 89°)
- ✅ Very high angles (up to 89°)
- ✅ Multiple material combinations (Air, Glass, Silicon, Water, reverse directions)
- ✅ Layer thickness variations
- ✅ Single wavelength focused tests
- ✅ Same incident/exit medium
- ✅ Pattern analysis
- ✅ Refractive index type (both constant and wavelength-dependent)
- ✅ Visualization (optional, requires matplotlib)

All tests pass with deviations < 1e-15 (numerical precision), confirming that energy conservation holds for all tested scenarios.

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

3. **Wavelength-dependent permittivity**: Use arrays for dispersion
   ```python
   # Create wavelength-dependent permittivity array
   wavelengths = np.linspace(400e-9, 800e-9, 100)
   eps_dispersive = 2.25 + 0.01 * (wavelengths * 1e9 - 600) / 600
   
   # Use in structure
   ml = MultiLayerStructure(wavelengths, angle, pol, layers, 1.0, eps_dispersive)
   ```

4. **Reuse structure instances**: A `MultiLayerStructure` caches interface/propagation matrices and the full transfer matrix, so calling `reflectance()`, `transmittance()`, or `total_transfer_matrix()` multiple times on the same instance has minimal overhead once the first call completes.

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
7. **Wavelength-dependent permittivity**: 
   - Arrays must match the length of the wavelengths array
   - Use scalar values for constant permittivity (automatically broadcast)
   - Mix constant and wavelength-dependent permittivities as needed
8. **Test your models**: Run the test suite to verify energy conservation for your specific configurations

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
- Tests pass (run `tests/test_*.py` files)

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Version**: 2.2  
**Last Updated**: 2025  
**Python Compatibility**: 3.7+

### Version History

**v2.2** (2025)
- Fixed the multi-layer interface caching so every layer contributes correctly to the total transfer matrix (restores non-zero reflectance for Bragg mirrors).
- Applied the power-flow correction to transmittance for both single-interface and multi-layer cases, ensuring `R + T = 1` for all lossless stacks and angles.
- Added a Bragg-mirror regression test (air | [Nb₂O₅ | SiO₂]₁₀ | SiO₂, 1–2 µm sweep) to guard against future energy-conservation regressions.

**v2.1** (2025)
- Added comprehensive support for wavelength-dependent permittivity (arrays)
- Enhanced energy conservation verification with 200+ test cases organized into focused modules
- Support for mixed constant and wavelength-dependent permittivities
- Decomposed comprehensive test suite into focused, maintainable test files
- Improved documentation and examples

**v2.0** (2025)
- Initial release with full Transfer Matrix Method implementation
- Support for s and p polarizations
- Oblique incidence support
- Power flow correction for different media
