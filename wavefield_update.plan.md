# WaveField Update Plan v2: k_vectors with kx and kz components

## Overview

Update the `WaveField` dataclass to replace the simple `k_vectors` array with a 2-column array containing both tangential (kx) and normal (kz) wave vector components. Also rename `field` to `amplitude` and add spatial field calculation methods.

## Completed Changes

### 1. Attribute Renames

**Before:**
```python
@dataclass
class WaveField:
    k_vectors: np.ndarray  # Shape (N,)
    field: np.ndarray      # Shape (N, 2)
```

**After:**
```python
@dataclass
class WaveField:
    k_vectors: np.ndarray  # Shape (N, 2) - [kx, kz]
    amplitude: np.ndarray  # Shape (N, 2)
```

### 2. Updated Data Structure

**k_vectors**: `np.ndarray`, shape `(N, 2)`
- Column 0: kx (tangential component, 1/m)
- Column 1: kz (normal component, 1/m)
- Must be 2D array with exactly 2 columns
- Can be real or complex (for lossy materials)

**amplitude**: `np.ndarray`, shape `(N, 2)`
- Column 0: forward propagating amplitude
- Column 1: backward propagating amplitude
- Must be complex dtype

### 3. Updated Properties

**Before:**
```python
@property
def forward(self) -> np.ndarray:
    return self.field[:, 0]

@property
def backward(self) -> np.ndarray:
    return self.field[:, 1]
```

**After:**
```python
@property
def amp_forward(self) -> np.ndarray:
    """Forward propagating amplitude, shape (N,)."""
    return self.amplitude[:, 0]

@property
def amp_backward(self) -> np.ndarray:
    """Backward propagating amplitude, shape (N,)."""
    return self.amplitude[:, 1]
```

### 4. Updated Validation (`__post_init__`)

**Validation Requirements:**

1. **k_vectors validation:**
   - Must be 2D array
   - Must have exactly 2 columns (kx, kz)
   - Must have at least 1 row
   - Raises `ValueError` if validation fails

2. **amplitude validation:**
   - Must be 2D array
   - Must have exactly 2 columns
   - Number of rows must match k_vectors
   - Converted to complex dtype if needed

**Error Messages:**
- "k_vectors must be 2D array, got shape ..."
- "k_vectors must have 2 columns (kx, kz), got ..."
- "k_vectors rows (...) must match amplitude rows (...)"

### 5. New Spatial Field Calculation Methods

Added two new methods to calculate field values at spatial coordinates (x, z):

#### `field_forward(x, z) -> np.ndarray`

Calculates forward propagating field at spatial coordinates.

**Formula:**
```
field_forward = amp_forward * exp(1j * kx * x + 1j * kz * z)
```

**Parameters:**
- `x` (float): Tangential spatial coordinate (meters)
- `z` (float): Normal spatial coordinate (meters)

**Returns:**
- `np.ndarray`: Forward field at coordinates (x, z), shape (N,)

#### `field_backward(x, z) -> np.ndarray`

Calculates backward propagating field at spatial coordinates.

**Formula:**
```
field_backward = amp_backward * exp(1j * kx * x - 1j * kz * z)
```

**Parameters:**
- `x` (float): Tangential spatial coordinate (meters)
- `z` (float): Normal spatial coordinate (meters)

**Returns:**
- `np.ndarray`: Backward field at coordinates (x, z), shape (N,)

### 6. Updated Docstring

**Updated parameter documentation:**

```python
Parameters
----------
k_vectors : array_like
    Wave vector array, shape (N, 2)
    Column 0: kx (tangential component)
    Column 1: kz (normal component)
amplitude : array_like
    Complex field array, shape (N, 2)
    Column 0: forward propagating component
    Column 1: backward propagating component
```

**Updated example:**

```python
>>> import numpy as np
>>> kx = np.array([1.0e7, 1.2e7, 1.4e7])
>>> kz = np.array([2.0e7, 2.2e7, 2.4e7])
>>> k_vectors = np.column_stack([kx, kz])
>>> forward_amp = np.array([1.0+0j, 0.9+0.1j, 0.8+0.2j])
>>> backward_amp = np.array([0.1+0j, 0.15+0.05j, 0.2+0.1j])
>>> amplitude_data = np.column_stack([forward_amp, backward_amp])
>>> wave_field = WaveField(k_vectors, amplitude_data)
>>> print(wave_field.amp_forward)   # Forward amplitude
>>> print(wave_field.amp_backward)  # Backward amplitude
```

### 7. Usage Examples

**Creating a WaveField object:**

```python
import numpy as np
from TMatrix import WaveField

# Create k_vectors (kx, kz for each wavelength)
kx = np.array([1.0e7, 1.2e7, 1.4e7])  # Tangential components
kz = np.array([2.0e7, 2.2e7, 2.4e7])  # Normal components
k_vectors = np.column_stack([kx, kz])

# Create amplitude data
forward_amp = np.array([1.0+0j, 0.9+0.1j, 0.8+0.2j])
backward_amp = np.array([0.1+0j, 0.15+0.05j, 0.2+0.1j])
amplitude = np.column_stack([forward_amp, backward_amp])

# Create WaveField
wave_field = WaveField(k_vectors, amplitude)

# Access amplitudes
print(wave_field.amp_forward)   # Shape: (3,)
print(wave_field.amp_backward)  # Shape: (3,)

# Calculate spatial fields
x = 0.5e-6  # 0.5 micrometers
z = 0.1e-6  # 0.1 micrometers
forward_field = wave_field.field_forward(x, z)
backward_field = wave_field.field_backward(x, z)
```

## Key Changes Summary

| Aspect | Before | After |
|--------|--------|-------|
| **k_vectors shape** | (N,) | (N, 2) - [kx, kz] |
| **Field attribute** | `field` | `amplitude` |
| **Forward property** | `forward` | `amp_forward` |
| **Backward property** | `backward` | `amp_backward` |
| **Spatial calculation** | None | `field_forward()`, `field_backward()` |
| **Validation** | 1D k_vectors | 2D with 2 columns |

## Physical Interpretation

The new structure allows for more physical representation:

- **kx** (tangential component): Conserved across interfaces, depends on incident angle
- **kz** (normal component): Changes with medium, determines phase accumulation
- **Spatial fields**: Full electromagnetic field at any (x, z) coordinate

## Benefits

1. **More physical**: Separates tangential and normal wave vector components
2. **Spatial awareness**: Can calculate fields at any point in space
3. **Better integration**: Aligns with transfer matrix physics
4. **Flexible**: Supports both lossless and lossy materials
5. **Vectorized**: Efficient calculation for wavelength arrays

## Implementation Status

✅ **Completed:**
- [x] Rename `field` → `amplitude`
- [x] Change `k_vectors` shape (N,) → (N, 2)
- [x] Rename `forward` → `amp_forward`
- [x] Rename `backward` → `amp_backward`
- [x] Add `field_forward(x, z)` method
- [x] Add `field_backward(x, z)` method
- [x] Update validation logic
- [x] Update docstrings and examples
- [x] Update error messages

## Files Modified

- `/Users/luqian/ScientificAI/TransferMatrix/TMatrix.py` - WaveField class (lines 10-163)

## Breaking Changes

⚠️ **Note:** This is a breaking change for existing code:

1. **Field access**: `wave.field` → `wave.amplitude`
2. **Property names**: `wave.forward` → `wave.amp_forward`, `wave.backward` → `wave.amp_backward`
3. **k_vectors shape**: Now requires (N, 2) instead of (N,)

Code using the old WaveField API will need to be updated to use the new attribute and property names.

## Migration Guide

**Old code:**
```python
wave = WaveField(k_vectors, field_data)
forward = wave.forward
backward = wave.backward
```

**New code:**
```python
wave = WaveField(k_vectors, amplitude)
forward = wave.amp_forward
backward = wave.amp_backward
# Or calculate spatial fields
forward_field = wave.field_forward(x, z)
backward_field = wave.field_backward(x, z)
```

## Future Enhancements

Potential additions (not yet implemented):

- [ ] Automatic k_vector generation from wavelength and angle
- [ ] Bulk spatial field calculation for multiple points
- [ ] Field visualization methods
- [ ] Integration with SingleInterfaceTMatrix/LayerPropagationMatrix
- [ ] Poynting vector calculation
- [ ] Energy flux computation
