"""
Common setup and material definitions for energy conservation tests.
"""

import numpy as np

# Test tolerance
tolerance = 1e-10

# Wavelength arrays
wavelengths = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
wavelength_single = np.array([600e-9])  # Single wavelength for focused tests
n_wavelengths = len(wavelengths)

# Constant permittivities
eps_air_const = 1.0
eps_glass_const = 2.25
eps_silica_const = 2.13
eps_titanium_const = 5.76
eps_silicon_const = 12.25  # n = 3.5
eps_water_const = 1.77  # n ≈ 1.33

# Wavelength-dependent permittivity arrays
eps_air_array = np.ones(n_wavelengths)  # Air is constant
eps_glass_array = 2.25 + 0.01 * (wavelengths * 1e9 - 600) / 600  # Slight dispersion
eps_silica_array = 2.13 + 0.005 * (wavelengths * 1e9 - 600) / 600
eps_titanium_array = 5.76 + 0.02 * (wavelengths * 1e9 - 600) / 600
eps_silicon_array = 12.25 + 0.05 * (wavelengths * 1e9 - 600) / 600
eps_water_array = 1.77 + 0.002 * (wavelengths * 1e9 - 600) / 600

# Refractive indices
n_glass_const = 1.5
n_silica_const = 1.46
n_glass_array = 1.5 + 0.01 * (wavelengths * 1e9 - 600) / 600

