#!/usr/bin/env python
"""
Silicon Refractive Index Fitting and Evaluation.

Provides a simple interface to evaluate Silicon's refractive index (n and kappa)
and permittivity (epsilon) at any wavelength based on experimental data.
"""

import numpy as np
from .fit_refractive_index import RefractiveIndexFitter, load_data


class SiliconRefractiveIndex:
    """Class to evaluate Silicon refractive index from fitted data."""
    
    def __init__(self, data_file='Si_exp_RI_INME_RAS.txt', fit_method='polynomial'):
        """
        Initialize with data file and fitting method.
        
        Parameters
        ----------
        data_file : str
            Path to the refractive index data file
        fit_method : str
            Fitting method: 'polynomial', 'spline', or 'interpolation'
        """
        import os
        # If relative path, make it relative to this module's directory
        if not os.path.isabs(data_file):
            module_dir = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(module_dir, data_file)
        wavelength, n, kappa = load_data(data_file)
        self.fitter = RefractiveIndexFitter(wavelength, n, kappa)
        
        # Perform fitting
        if fit_method == 'polynomial':
            self.fitter.fit_polynomial(n_degree=6, kappa_degree=5)
        elif fit_method == 'spline':
            self.fitter.fit_spline(n_s=0.1, kappa_s=0.1)
        elif fit_method == 'interpolation':
            self.fitter.fit_interpolation(kind='cubic')
        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")
        
        self.wavelength_range = (wavelength.min(), wavelength.max())
        
    def n(self, wavelength):
        """
        Get real part of refractive index.
        
        Parameters
        ----------
        wavelength : float or array
            Wavelength(s) in nanometers
            
        Returns
        -------
        float or array
            Refractive index n
        """
        return self.fitter.evaluate_n(wavelength)
    
    def kappa(self, wavelength):
        """
        Get imaginary part of refractive index (extinction coefficient).
        
        Parameters
        ----------
        wavelength : float or array
            Wavelength(s) in nanometers
            
        Returns
        -------
        float or array
            Extinction coefficient kappa
        """
        return self.fitter.evaluate_kappa(wavelength)
    
    def epsilon(self, wavelength):
        """
        Get complex permittivity: epsilon = (n + i*kappa)².
        
        Parameters
        ----------
        wavelength : float or array
            Wavelength(s) in nanometers
            
        Returns
        -------
        complex or array of complex
            Complex permittivity
        """
        return self.fitter.evaluate_epsilon(wavelength)
    
    def __call__(self, wavelength):
        """Convenience method: returns (n, kappa, epsilon) tuple."""
        n_vals = self.n(wavelength)
        kappa_vals = self.kappa(wavelength)
        eps_vals = self.epsilon(wavelength)
        return n_vals, kappa_vals, eps_vals


# Create a default instance for easy use
_default_si_ri = None

def get_silicon_ri(fit_method='polynomial'):
    """
    Get a Silicon refractive index evaluator.
    
    Parameters
    ----------
    fit_method : str
        Fitting method: 'polynomial', 'spline', or 'interpolation'
        
    Returns
    -------
    SiliconRefractiveIndex
        Evaluator instance
    """
    global _default_si_ri
    if _default_si_ri is None or _default_si_ri.fitter.fit_method != fit_method:
        _default_si_ri = SiliconRefractiveIndex(fit_method=fit_method)
    return _default_si_ri


# Example usage
if __name__ == '__main__':
    # Create evaluator
    si = SiliconRefractiveIndex(fit_method='polynomial')
    
    # Evaluate at specific wavelengths
    wavelengths = np.array([400, 500, 600, 700, 800, 900])  # nm
    
    print("=" * 80)
    print("Silicon Refractive Index Evaluation")
    print("=" * 80)
    print(f"\nWavelength range: {si.wavelength_range[0]:.2f} - {si.wavelength_range[1]:.2f} nm")
    print(f"\n{'Wavelength (nm)':<15} {'n':<10} {'κ':<12} {'ε_real':<12} {'ε_imag':<12}")
    print("-" * 80)
    
    for wl in wavelengths:
        n_val = si.n(wl)
        kappa_val = si.kappa(wl)
        eps_val = si.epsilon(wl)
        print(f"{wl:<15.1f} {n_val:<10.4f} {kappa_val:<12.6f} "
              f"{np.real(eps_val):<12.4f} {np.imag(eps_val):<12.4f}")
    
    # Example: Use with TMatrix
    print("\n" + "=" * 80)
    print("Example: Using with TMatrix")
    print("=" * 80)
    
    try:
        from TMatrix import Layer, MultiLayerStructure
        
        # Create a Silicon layer using the fitted permittivity
        test_wavelengths = np.array([500e-9, 600e-9, 700e-9])  # meters
        
        # Convert to nm for refractive index lookup
        wl_nm = test_wavelengths * 1e9
        eps_si = si.epsilon(wl_nm)
        
        print(f"\nCreating Silicon layer with fitted permittivity:")
        print(f"  Wavelengths: {test_wavelengths*1e9} nm")
        print(f"  Permittivity: {eps_si}")
        
        # Create layer (using first wavelength's permittivity as example)
        # Note: TMatrix expects wavelength-dependent permittivity
        layers = [Layer(
            thickness=100e-9,
            optical_property={'type': 'permittivity', 'value': eps_si[0]}  # Use first value
        )]
        
        ml = MultiLayerStructure(
            wavelengths=test_wavelengths,
            angle_degrees=0.0,
            polarization='s',
            layers=layers,
            eps_incident=1.0,
            eps_exit=1.0
        )
        
        R = ml.reflectance()
        T = ml.transmittance()
        
        print(f"\nReflectance: {R}")
        print(f"Transmittance: {T}")
        print(f"R + T: {R + T}")
        
    except ImportError:
        print("\nTMatrix not available for this example.")

