"""
Common setup and material definitions for energy conservation tests.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import TMatrix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np

# Test tolerance
tolerance = 1e-10

# Wavelength arrays
wavelengths = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
wavelength_single = np.array([600e-9])  # Single wavelength for focused tests
n_wavelengths = len(wavelengths)

# Constant permittivities
eps_air_const = 1.0
eps_glass_const = 1.47**2
eps_silica_const = 1.47**2
eps_titanium_const = 5.76
eps_silicon_const = 12.25  # n = 3.5
eps_water_const = 1.77  # n ≈ 1.33
eps_Nb2O5_const = 2.24**2

# Wavelength-dependent permittivity arrays
eps_air_array = np.ones(n_wavelengths)  # Air is constant
eps_glass_array = 2.25 + 0.01 * (wavelengths * 1e9 - 600) / 600  # Slight dispersion
eps_silica_array = 2.13 + 0.005 * (wavelengths * 1e9 - 600) / 600
eps_titanium_array = 5.76 + 0.02 * (wavelengths * 1e9 - 600) / 600
eps_silicon_array = 12.25 + 0.05 * (wavelengths * 1e9 - 600) / 600
eps_water_array = 1.77 + 0.002 * (wavelengths * 1e9 - 600) / 600
eps_Nb2O5_array = eps_Nb2O5_const * np.ones(n_wavelengths)

# Refractive indices
n_glass_const = 1.47
n_silica_const = 1.47
n_glass_array = 1.47 + 0.01 * (wavelengths * 1e9 - 600) / 600

# Impedance of free space (Z_0 = sqrt(μ_0/ε_0) ≈ 376.73 Ω)
Z_0_FREE_SPACE = 376.730313668  # Ohms


# Function to calculate impedance from permittivity
# For non-magnetic materials: Z = Z_0 / sqrt(eps)
# where eps is the relative permittivity (ε_r)
def calculate_impedance(eps):
    """
    Calculate impedance from permittivity.

    Parameters:
    -----------
    eps : float or np.ndarray
        Relative permittivity (ε_r)

    Returns:
    --------
    float or np.ndarray
        Impedance in Ohms: Z = Z_0 / sqrt(eps)
    """
    # Use real part of permittivity for impedance calculation
    eps_real = np.real(eps) if isinstance(eps, np.ndarray) else np.real(eps)
    return Z_0_FREE_SPACE / np.sqrt(eps_real + 0j)


# Input electric field in Volts per meter
E_in = 5.309e7  # 53.09 MV/m
