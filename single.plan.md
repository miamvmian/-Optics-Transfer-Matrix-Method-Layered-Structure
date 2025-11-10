# Transfer Matrix Method for Wavelength Arrays

## Overview

This document outlines the implementation of transfer matrix methods that compute 2x2 transfer matrices across multiple wavelengths with a fixed incident angle. The implementation includes:

1. **SingleInterfaceTMatrix**: For single interface calculations
2. **LayerPropagationMatrix**: For phase accumulation through a single layer

Both classes support s-polarization (TE) and p-polarization (TM).

---

## Class Structure

### `SingleInterfaceTMatrix`

The main class that handles single interface transfer matrix calculations with wavelength array support.

#### Constructor

```python
__init__(self, lda: Union[float, np.ndarray], theta: float, polarization: str,
         eps_in: Union[float, np.ndarray], eps_out: Union[float, np.ndarray])
```

**Parameters:**

- `lda`: Wavelength array in meters, shape (N,) or scalar
- `theta`: Incident angle in degrees (single value, not array)
- `polarization`: Polarization mode ('s' for TE, 'p' for TM)
- `eps_in`: Input medium permittivity (scalar or shape (N,))
- `eps_out`: Output medium permittivity (scalar or shape (N,))

**Attributes:**

- `lda`: Wavelength array in meters, shape (N,)
- `theta`: Incident angle in radians
- `polarization`: Polarization mode
- `eps_in`: Input medium permittivity, shape (N,)
- `eps_out`: Output medium permittivity, shape (N,)
- `k0`: Free space wave vectors, shape (N,)
- `kx`: Tangential wave vectors, shape (N,)
- `kz_in`: Input medium z-components, shape (N,)
- `kz_out`: Output medium z-components, shape (N,)

#### Core Methods

##### `full_transfer_matrix() -> np.ndarray`

Calculates transfer matrices for the single interface at all wavelengths.

- **Returns:** Transfer matrices, shape (N, 2, 2) where N is number of wavelengths
- **Formula:** T = F_out * F_in^{-1}

##### `k_vectors() -> Dict[str, np.ndarray]`

Returns all calculated wave vectors.

- **Returns:** Dictionary containing k0, kx, kz_in, kz_out arrays

#### Internal Methods

##### `_validate_inputs(lda, theta, polarization, eps_in, eps_out)`

Validates input parameters including:

- Wavelength array must be 1D and positive
- Angle must be in range [0, 90) degrees
- Polarization must be 's' or 'p'
- Permittivities must be scalar or 1D arrays

##### `_calculate_kz(eps: np.ndarray) -> np.ndarray`

Calculates z-component of wave vector with proper branch selection.

- **Formula:** kz = sqrt(k0^2 * eps - kx^2 + i·1e-30)

##### `_half_transfer_matrix_s(eps: np.ndarray, kz: np.ndarray) -> np.ndarray`

Half field transfer matrix at interface for s-polarization (TE mode).

- **Returns:** Half field transfer matrices, shape (N, 2, 2)

##### `_half_transfer_matrix_s_inv(eps: np.ndarray, kz: np.ndarray) -> np.ndarray`

Inverse half field transfer matrix at interface for s-polarization (TE mode).

- **Returns:** Inverse half field transfer matrices, shape (N, 2, 2)

##### `_half_transfer_matrix_p(eps: np.ndarray, kz: np.ndarray) -> np.ndarray`

Half field transfer matrix at interface for p-polarization (TM mode).

- **Returns:** Half field transfer matrices, shape (N, 2, 2)

##### `_half_transfer_matrix_p_inv(eps: np.ndarray, kz: np.ndarray) -> np.ndarray`

Inverse half field transfer matrix at interface for p-polarization (TM mode).

- **Returns:** Inverse half field transfer matrices, shape (N, 2, 2)

---

### `LayerPropagationMatrix`

A standalone class that computes the propagation (phase accumulation) matrix for wave propagation through a single layer at multiple wavelengths.

#### Constructor

```python
__init__(self, lda: Union[float, np.ndarray], theta: float,
         d: float, eps: Union[float, np.ndarray])
```

**Parameters:**

- `lda`: Wavelength array in meters, shape (N,) or scalar
- `theta`: Incident angle in degrees (single value, not array)
- `d`: Layer thickness in meters (scalar)
- `eps`: Layer permittivity (scalar or shape (N,))

**Attributes:**

- `lda`: Wavelength array in meters, shape (N,)
- `theta`: Incident angle in radians
- `d`: Layer thickness in meters
- `eps`: Layer permittivity, shape (N,)
- `k0`: Free space wave vectors, shape (N,)
- `kx`: Tangential wave vectors, shape (N,)
- `kz`: z-component wave vectors, shape (N,)

#### Core Methods

##### `propagation_matrix() -> np.ndarray`

Calculate the propagation matrix for all wavelengths.

- **Returns:** Propagation matrices, shape (N, 2, 2)
- **Formula:** P = [[exp(i*kz*d), 0], [0, exp(-i*kz*d)]]
- **Note:** Propagation matrix is polarization-independent

##### `k_vectors() -> Dict[str, np.ndarray]`

Returns all calculated wave vectors.

- **Returns:** Dictionary containing k0, kx, kz arrays

#### Internal Methods

##### `_validate_inputs(lda, theta, d, eps)`

Validates input parameters including:

- Wavelength array must be 1D and positive
- Angle must be in range [0, 90) degrees
- Layer thickness `d` must be non-negative scalar
- Permittivity must be scalar or 1D array

##### `_calculate_kz(eps: np.ndarray) -> np.ndarray`

Calculates z-component of wave vector with proper branch selection.

- **Formula:** kz = sqrt(k0^2 * eps - kx^2 + i·1e-30)

---

## Implementation Details

### Wave Vector Calculations

- `k0 = 2π/lda` (per-wavelength)
- `kx = k0 * sqrt(eps_in) * sin(theta)` (per-wavelength)
- `kz_j = sqrt(k0^2 * eps_j - kx^2 + i·1e-30)` for each medium/layer

### Transfer Matrix Calculation

#### Single Interface (no layer thickness)

- **s-polarization:** T = F_out_s * F_in_s_inv
- **p-polarization:** T = F_out_p * F_in_p_inv

#### Layer Propagation (phase accumulation)

- **Both polarizations:** P = [[exp(i*kz*d), 0], [0, exp(-i*kz*d)]]

#### Full Single Layer (manual combination)

Users can combine interface and propagation matrices:

- **Formula:** T_full = F_out @ P @ F_in_inv

### Vectorization

- All calculations are vectorized to handle wavelength arrays efficiently
- Uses `np.matmul` for batch matrix multiplication
- Broadcasting handles scalar inputs automatically

---

## Validation Rules

1. **Wavelength validation:**
   - Must be 1D array
   - All values must be positive

2. **Angle validation:**
   - Must be in range [0, 90) degrees
   - Exactly 90° is not supported (grazing incidence)
   - Warning for angles near 90°

3. **Polarization validation:**
   - Must be 's' or 'p'

4. **Permittivity validation:**
   - Must be scalar or 1D array
   - If array, length must match wavelength array length

5. **Layer thickness validation (LayerPropagationMatrix only):**
   - Must be non-negative scalar

---

## Usage Examples

### Example 1: Single Interface

```python
import numpy as np
from TMatrix import SingleInterfaceTMatrix

# Define wavelength array
lda = np.array([500e-9, 600e-9, 700e-9])  # 3 wavelengths

# Define parameters
theta = 30.0  # 30 degrees
eps_in = 1.0  # Air
eps_out = 2.25  # Glass (constant)

# Create calculator
tm = SingleInterfaceTMatrix(lda, theta, 's', eps_in, eps_out)

# Calculate transfer matrices
T_matrices = tm.full_transfer_matrix()  # Shape (3, 2, 2)

# Get wave vectors
k_vectors = tm.k_vectors()
```

### Example 2: Layer Propagation

```python
import numpy as np
from TMatrix import LayerPropagationMatrix

# Define parameters
lda = np.array([500e-9, 600e-9, 700e-9])
theta = 30.0
d = 100e-9  # 100 nm layer thickness
eps = 2.25  # Glass

# Calculate propagation matrix
pm = LayerPropagationMatrix(lda, theta, d, eps)
P = pm.propagation_matrix()  # Shape (3, 2, 2)

# Get wave vectors
k_vectors = pm.k_vectors()
```

### Example 3: Manual Combination for Full Layer

```python
import numpy as np
from TMatrix import SingleInterfaceTMatrix, LayerPropagationMatrix

# Define parameters
lda = np.array([500e-9, 600e-9, 700e-9])
theta = 30.0
d = 100e-9  # Layer thickness
eps_in = 1.0  # Air
eps_layer = 2.25  # Glass layer
eps_out = 1.0  # Air

# Calculate interface matrices
tm_in = SingleInterfaceTMatrix(lda, theta, 's', eps_in, eps_layer)
tm_out = SingleInterfaceTMatrix(lda, theta, 's', eps_layer, eps_out)

# Calculate propagation matrix
pm = LayerPropagationMatrix(lda, theta, d, eps_layer)

# Get matrices
F_in_inv = tm_in.full_transfer_matrix()
P = pm.propagation_matrix()
F_out = tm_out.full_transfer_matrix()

# Combine manually: T_full = F_out @ P @ F_in_inv
# (User implements combination)
T_full = np.matmul(np.matmul(F_out, P), F_in_inv)
```

---

## Non-Goals (Deferred)

- Multi-layer stacks (automatic composition)
- Start/end layer special handling
- Reflectance/transmittance calculations
- Angle arrays
- Parallelization and caching
- Automatic combination of interface and propagation matrices
