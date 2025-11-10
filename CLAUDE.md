# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python implementation of the Transfer Matrix Method for optical calculations, focusing on single interfaces and single layers. The implementation is optimized for wavelength array operations with vectorized calculations.

## Core Implementation (TMatrix.py)

**Location**: `/Users/luqian/ScientificAI/TransferMatrix/TMatrix.py`

This module provides three main classes for optical modeling:

### 1. WaveField (Dataclass)
A data structure for storing electromagnetic field information.

**Purpose**: Store forward and backward propagating wave components

- **Parameters**:
  - `k_vectors`: Wave vector array, shape (N,) - can be real or complex
  - `field`: Complex field array, shape (N, 2) - column 0 is forward, column 1 is backward
- **Properties**:
  - `forward`: Forward propagating component, shape (N,)
  - `backward`: Backward propagating component, shape (N,)
  - `n_points`: Number of wave vector points
- **Validation**: Comprehensive input validation with shape checking
- **Use case**: Manual field construction for advanced applications

### 2. SingleInterfaceTMatrix (Class)
Calculates 2×2 transfer matrices for single interfaces across multiple wavelengths.

**Purpose**: Compute transfer matrices for interface between two media

- **Parameters**:
  - `lda`: Wavelength array in meters, shape (N,) - can be scalar or array
  - `theta`: Incident angle in degrees (scalar, single value)
  - `polarization`: 's' for TE, 'p' for TM
  - `eps_in`: Input medium permittivity (scalar or array)
  - `eps_out`: Output medium permittivity (scalar or array)
- **Key Attributes**:
  - `lda`: Wavelength array, shape (N,)
  - `k0`: Free space wave vectors, shape (N,)
  - `kx`: Tangential wave vectors, shape (N,)
  - `kz_in`, `kz_out`: z-components in each medium, shape (N,)
- **Main Methods**:
  - `full_transfer_matrix()`: Returns transfer matrices, shape (N, 2, 2)
  - `k_vectors()`: Returns dict with all wave vectors
- **Internal Methods**:
  - `_half_transfer_matrix_s()`: s-polarization field matrix
  - `_half_transfer_matrix_s_inv()`: s-polarization inverse field matrix
  - `_half_transfer_matrix_p()`: p-polarization field matrix
  - `_half_transfer_matrix_p_inv()`: p-polarization inverse field matrix
  - `_calculate_kz()`: Computes z-component of wave vector

**Transfer Matrix Formula**: T = F_out @ F_in^(-1)

### 3. LayerPropagationMatrix (Class)
Computes 2×2 propagation matrices for phase accumulation through a single layer.

**Purpose**: Calculate phase accumulation through a layer

- **Parameters**:
  - `lda`: Wavelength array in meters, shape (N,)
  - `theta`: Incident angle in degrees (scalar)
  - `d`: Layer thickness in meters (scalar)
  - `eps`: Layer permittivity (scalar or array)
- **Key Attributes**:
  - `lda`: Wavelength array, shape (N,)
  - `kz`: z-component wave vectors in layer, shape (N,)
- **Main Methods**:
  - `propagation_matrix()`: Returns propagation matrices, shape (N, 2, 2)
  - `k_vectors()`: Returns dict with all wave vectors

**Propagation Matrix Formula**: P = [[exp(i*kz*d), 0], [0, exp(-i*kz*d)]]

## Input Conventions

### Critical Requirements
- **Wavelengths**: Must be in meters (NOT nm or μm)
- **Angles**: Input in degrees, converted to radians internally
- **Incident angle**: MUST be scalar (not array), range [0, 90)
- **Exactly 90°**: Not supported (raises ValueError)
- **Units**: All lengths in meters

### Data Types
- **Permittivity**: Supports both real (dielectric) and complex (lossy materials)
- **Broadcasting**: Scalar inputs automatically broadcast to wavelength array length
- **Complex numbers**: Full support for lossy materials (metals)

### Polarization
- `'s'` or `'S'`: s-polarization (TE mode)
  - Electric field perpendicular to plane of incidence
  - Field matrix: F = [[1, 1], [-kz/k0, kz/k0]]
- `'p'` or `'P'`: p-polarization (TM mode)
  - Magnetic field perpendicular to plane of incidence
  - Field matrix: F = [[kz/(k0*eps), -kz/(k0*eps)], [1, 1]]

## Usage Examples

### Single Interface
```python
import numpy as np
from TMatrix import SingleInterfaceTMatrix

# Define parameters
lda = np.array([500e-9, 600e-9, 700e-9])  # wavelengths in meters
theta = 30.0  # degrees
eps_in = 1.0  # air
eps_out = 2.25  # glass

# Calculate transfer matrix
tm = SingleInterfaceTMatrix(lda, theta, 's', eps_in, eps_out)
T = tm.full_transfer_matrix()  # Shape: (3, 2, 2)

# Get wave vectors
k_vecs = tm.k_vectors()
kz_out = k_vecs['kz_out']
```

### Layer Propagation
```python
from TMatrix import LayerPropagationMatrix

d = 100e-9  # 100 nm layer
eps = 2.25  # glass

pm = LayerPropagationMatrix(lda, theta, d, eps)
P = pm.propagation_matrix()  # Shape: (3, 2, 2)
```

### Manual Combination for Full Layer
```python
# air | glass | air
lda = np.array([500e-9, 600e-9, 700e-9])
theta = 30.0
d = 100e-9
eps_in = 1.0
eps_layer = 2.25
eps_out = 1.0

# Calculate interface matrices
tm_in = SingleInterfaceTMatrix(lda, theta, 's', eps_in, eps_layer)
tm_out = SingleInterfaceTMatrix(lda, theta, 's', eps_layer, eps_out)

# Calculate propagation
pm = LayerPropagationMatrix(lda, theta, d, eps_layer)

# Get matrices
F_in_inv = tm_in.full_transfer_matrix()
P = pm.propagation_matrix()
F_out = tm_out.full_transfer_matrix()

# Combine: T_full = F_out @ P @ F_in_inv
T_full = np.matmul(np.matmul(F_out, P), F_in_inv)
```

### Complex Permittivity (Lossy Material)
```python
# Air to gold interface
eps_in = 1.0  # air
eps_out = -25 + 1.5j  # gold (complex)

tm = SingleInterfaceTMatrix(wavelengths, theta, 'p', eps_in, eps_out)
T = tm.full_transfer_matrix()
```

## Key Physical Formulas

### Wave Vectors
- `k0 = 2π/λ` (free space wave vector)
- `kx = k0 * sqrt(eps_in) * sin(θ)` (tangential component, conserved across interface)
- `kz = sqrt(k0² * eps - kx²)` (normal component, with proper branch selection)

### Branch Selection
The `_calculate_kz()` method adds a small imaginary part (1e-30) for numerical stability:
```python
kz = np.sqrt(kz_squared + 1j * 1e-30)
```

### Field Matrices

**s-polarization**:
- F = [[1, 1], [-kz/k0, kz/k0]]
- F^(-1) = [[0.5, -0.5*k0/kz], [0.5, 0.5*k0/kz]]

**p-polarization**:
- F = [[kz/(k0*eps), -kz/(k0*eps)], [1, 1]]
- F^(-1) = [[0.5*k0*eps/kz, 0.5], [-0.5*k0*eps/kz, 0.5]]

## Input Validation

The classes implement comprehensive validation:

1. **Wavelengths**: Must be 1D, all positive
2. **Angles**: Must not be exactly 90° (ValueError), warns if outside [0, 90)
3. **Permittivity**: Must be scalar or 1D, length must match wavelengths
4. **Polarization**: Must be 's' or 'p'
5. **Layer thickness**: Must be non-negative

## Numerical Stability

- Small imaginary addition (1e-30) prevents branch cut issues
- Proper branch selection for kz calculation
- Broadcasting prevents dimension mismatches
- Vectorized operations for efficiency

## Performance

- **Vectorized**: All operations work on wavelength arrays
- **Single calculation**: ~0.01 ms
- **1000 wavelengths**: ~1 ms (vectorized)
- **Complex permittivities**: ~2× slower than real
- **Best practice**: Pass wavelength arrays instead of looping

## Error Handling

Common errors and solutions:

```python
# Negative wavelength
tm = SingleInterfaceTMatrix(-1e-6, 0, 's', 1, 2.25)
# ValueError: All wavelengths must be positive

# 90 degree angle
tm = SingleInterfaceTMatrix(1e-6, 90, 's', 1, 2.25)
# ValueError: Incident angle of exactly 90° is not supported

# Mismatched array sizes
lda = np.array([500e-9, 600e-9])
eps_out = np.array([2.25, 2.25, 3.0])  # 3 elements vs 2 wavelengths
tm = SingleInterfaceTMatrix(lda, 0, 's', 1, eps_out)
# ValueError: eps_out length must match wavelength array length
```

## Design Patterns

1. **Broadcasting**: Scalar eps values automatically broadcast to wavelength length
2. **Type Conversion**: All inputs converted to numpy arrays
3. **Private Methods**: Internal calculations prefixed with `_`
4. **Property Access**: WaveField uses @property for forward/backward components
5. **Docstrings**: NumPy-style with Parameters, Returns, Examples sections
6. **Type Hints**: Full type hints using Union and np.ndarray

## Limitations

1. **Single interface/layer only**: No built-in multi-layer support
2. **Scalar angle**: Incident angle must be scalar, not array
3. **Linear media**: No nonlinear effects
4. **Planar geometry**: Assumes infinite flat interfaces
5. **Grazing angles**: Degraded accuracy near 90°

## Key Implementation Details

### Wave Vector Calculation
```python
# Tangential component (conserved)
kx = k0 * sqrt(eps_in) * sin(theta)

# Normal component (changes with medium)
kz_squared = k0**2 * eps - kx**2
kz = sqrt(kz_squared + 1j * 1e-30)  # Small imaginary part for stability
```

### Vectorized Operations
- All matrix operations use `np.matmul` for broadcasting
- Element-wise operations are naturally vectorized
- Avoid explicit loops over wavelengths

### Memory Management
- Matrices stored as (N, 2, 2) where N = number of wavelengths
- Pre-allocated arrays for efficiency
- No in-place modifications

## Common Pitfalls

1. **Wrong units**: Using nm/μm instead of meters
2. **Array angle**: Trying to pass angle as array (not supported)
3. **Forgetting to extract**: Not calling `full_transfer_matrix()` or `propagation_matrix()`
4. **90° incidence**: Exactly 90° causes numerical issues
5. **Energy conservation**: For lossless media, R + T should be 1.0 (check with external code)

