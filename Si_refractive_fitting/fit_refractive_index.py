#!/usr/bin/env python
"""
Curve fitting for Silicon refractive index data.

Fits both real part (n) and imaginary part (kappa) as functions of wavelength.
Provides multiple fitting methods: polynomial, spline, and physical models.
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from scipy.optimize import curve_fit
    from scipy.interpolate import UnivariateSpline, interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Using numpy-only methods.")
import warnings
warnings.filterwarnings('ignore')


def load_data(filename):
    """Load refractive index data from file."""
    data = np.loadtxt(filename)
    wavelength = data[:, 0]  # nm
    n = data[:, 1]  # real part
    kappa = data[:, 2]  # imaginary part
    return wavelength, n, kappa


# ============================================================================
# Fitting Functions
# ============================================================================

def polynomial_fit(x, *coeffs):
    """Polynomial function for fitting."""
    return np.polyval(coeffs[::-1], x)


def sellmeier_n(wl, A, B, C, D=0, E=0):
    """
    Sellmeier equation for refractive index.
    n² = A + B*λ²/(λ² - C²) + D*λ²/(λ² - E²)
    """
    wl2 = wl**2
    n_sq = A + B * wl2 / (wl2 - C**2)
    if D != 0 and E != 0:
        n_sq += D * wl2 / (wl2 - E**2)
    return np.sqrt(np.abs(n_sq))


def cauchy_n(wl, A, B, C=0):
    """
    Cauchy equation for refractive index.
    n = A + B/λ² + C/λ⁴
    """
    return A + B / (wl**2) + C / (wl**4)


def exponential_kappa(wl, A, B, C):
    """Exponential decay for extinction coefficient."""
    return A * np.exp(-B * (wl - C))


def power_law_kappa(wl, A, B, C):
    """Power law for extinction coefficient."""
    return A * (wl - C)**(-B)


# ============================================================================
# Fitting Classes
# ============================================================================

class RefractiveIndexFitter:
    """Class to fit refractive index data."""
    
    def __init__(self, wavelength, n, kappa):
        self.wavelength = wavelength
        self.n = n
        self.kappa = kappa
        self.n_fit_func = None
        self.kappa_fit_func = None
        self.n_params = None
        self.kappa_params = None
        self.fit_method = None
        
    def fit_polynomial(self, n_degree=5, kappa_degree=5):
        """Fit using polynomials."""
        self.fit_method = 'polynomial'
        
        # Fit n
        n_coeffs = np.polyfit(self.wavelength, self.n, n_degree)
        self.n_fit_func = lambda wl: np.polyval(n_coeffs, wl)
        self.n_params = {'degree': n_degree, 'coeffs': n_coeffs}
        
        # Fit kappa (use log scale for better fit due to large dynamic range)
        log_kappa = np.log(self.kappa + 1e-10)
        kappa_coeffs = np.polyfit(self.wavelength, log_kappa, kappa_degree)
        self.kappa_fit_func = lambda wl: np.exp(np.polyval(kappa_coeffs, wl))
        self.kappa_params = {'degree': kappa_degree, 'coeffs': kappa_coeffs}
        
    def fit_spline(self, n_s=0.1, kappa_s=0.1):
        """Fit using splines."""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for spline fitting. Use polynomial or interpolation instead.")
        self.fit_method = 'spline'
        
        # Fit n
        n_spline = UnivariateSpline(self.wavelength, self.n, s=n_s)
        self.n_fit_func = n_spline
        self.n_params = {'s': n_s}
        
        # Fit kappa (use log scale)
        log_kappa = np.log(self.kappa + 1e-10)
        kappa_spline = UnivariateSpline(self.wavelength, log_kappa, s=kappa_s)
        self.kappa_fit_func = lambda wl: np.exp(kappa_spline(wl))
        self.kappa_params = {'s': kappa_s}
        
    def fit_interpolation(self, kind='cubic'):
        """Fit using interpolation."""
        if not HAS_SCIPY:
            # Use numpy interpolation as fallback
            self.fit_method = f'interpolation_{kind}_numpy'
            # For numpy, we'll use polynomial interpolation
            self.fit_polynomial(n_degree=min(10, len(self.wavelength)-1), 
                               kappa_degree=min(8, len(self.wavelength)-1))
            return
        
        self.fit_method = f'interpolation_{kind}'
        
        # Fit n
        self.n_fit_func = interp1d(self.wavelength, self.n, kind=kind, 
                                   bounds_error=False, fill_value='extrapolate')
        self.n_params = {'kind': kind}
        
        # Fit kappa
        self.kappa_fit_func = interp1d(self.wavelength, self.kappa, kind=kind,
                                       bounds_error=False, fill_value='extrapolate')
        self.kappa_params = {'kind': kind}
        
    def fit_sellmeier(self):
        """Fit n using Sellmeier equation."""
        if not HAS_SCIPY:
            print("Warning: scipy required for Sellmeier fit. Using polynomial instead.")
            self.fit_polynomial(n_degree=4)
            return
            
        self.fit_method = 'sellmeier'
        
        # Initial guess for Sellmeier parameters
        n_avg = np.mean(self.n)
        wl_center = np.mean(self.wavelength)
        p0_n = [n_avg**2 - 1, 1.0, wl_center * 0.5]
        
        try:
            popt_n, _ = curve_fit(sellmeier_n, self.wavelength, self.n, p0=p0_n,
                                  maxfev=5000)
            self.n_fit_func = lambda wl: sellmeier_n(wl, *popt_n)
            self.n_params = {'A': popt_n[0], 'B': popt_n[1], 'C': popt_n[2]}
        except:
            # Fallback to polynomial
            self.fit_polynomial(n_degree=4)
            return
            
        # Fit kappa with exponential
        log_kappa = np.log(self.kappa + 1e-10)
        kappa_coeffs = np.polyfit(self.wavelength, log_kappa, 3)
        self.kappa_fit_func = lambda wl: np.exp(np.polyval(kappa_coeffs, wl))
        self.kappa_params = {'type': 'exponential_poly'}
        
    def evaluate_n(self, wavelength):
        """Evaluate fitted n at given wavelengths."""
        if self.n_fit_func is None:
            raise ValueError("No fit has been performed yet.")
        return self.n_fit_func(wavelength)
    
    def evaluate_kappa(self, wavelength):
        """Evaluate fitted kappa at given wavelengths."""
        if self.kappa_fit_func is None:
            raise ValueError("No fit has been performed yet.")
        return self.kappa_fit_func(wavelength)
    
    def evaluate_epsilon(self, wavelength):
        """Evaluate complex permittivity: epsilon = (n + i*kappa)²."""
        n_vals = self.evaluate_n(wavelength)
        kappa_vals = self.evaluate_kappa(wavelength)
        return (n_vals + 1j * kappa_vals)**2
    
    def calculate_rms_error(self):
        """Calculate RMS error of the fit."""
        n_pred = self.evaluate_n(self.wavelength)
        kappa_pred = self.evaluate_kappa(self.wavelength)
        
        n_rms = np.sqrt(np.mean((n_pred - self.n)**2))
        kappa_rms = np.sqrt(np.mean((kappa_pred - self.kappa)**2))
        
        return {'n_rms': n_rms, 'kappa_rms': kappa_rms}


# ============================================================================
# Visualization
# ============================================================================

def plot_fits(wavelength, n, kappa, fitter, save_path=None):
    """Plot the data and fits."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate fine wavelength grid for smooth curves
    wl_fine = np.linspace(wavelength.min(), wavelength.max(), 1000)
    n_fit = fitter.evaluate_n(wl_fine)
    kappa_fit = fitter.evaluate_kappa(wl_fine)
    
    # Plot n
    axes[0, 0].plot(wavelength, n, 'o', markersize=3, alpha=0.6, 
                    label='Data', color='blue')
    axes[0, 0].plot(wl_fine, n_fit, '-', linewidth=2, 
                    label=f'Fit ({fitter.fit_method})', color='red')
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Refractive Index (n)')
    axes[0, 0].set_title('Real Part of Refractive Index')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot kappa (linear scale)
    axes[0, 1].plot(wavelength, kappa, 'o', markersize=3, alpha=0.6,
                     label='Data', color='blue')
    axes[0, 1].plot(wl_fine, kappa_fit, '-', linewidth=2,
                     label=f'Fit ({fitter.fit_method})', color='red')
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Extinction Coefficient (κ)')
    axes[0, 1].set_title('Imaginary Part of Refractive Index')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot kappa (log scale)
    axes[1, 0].semilogy(wavelength, kappa, 'o', markersize=3, alpha=0.6,
                        label='Data', color='blue')
    axes[1, 0].semilogy(wl_fine, kappa_fit, '-', linewidth=2,
                        label=f'Fit ({fitter.fit_method})', color='red')
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Extinction Coefficient (κ) [log scale]')
    axes[1, 0].set_title('Imaginary Part (Log Scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    # Plot residuals
    n_residual = fitter.evaluate_n(wavelength) - n
    kappa_residual = fitter.evaluate_kappa(wavelength) - kappa
    
    axes[1, 1].plot(wavelength, n_residual, 'o', markersize=3, alpha=0.6,
                    label='n residual', color='blue')
    axes[1, 1].plot(wavelength, kappa_residual, 's', markersize=3, alpha=0.6,
                    label='κ residual', color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Wavelength (nm)')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].set_title('Fit Residuals')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function to perform fitting."""
    # Load data
    filename = 'Si_exp_RI_INME_RAS.txt'
    wavelength, n, kappa = load_data(filename)
    
    print("=" * 80)
    print("Silicon Refractive Index Curve Fitting")
    print("=" * 80)
    print(f"\nData loaded: {len(wavelength)} points")
    print(f"Wavelength range: {wavelength.min():.2f} - {wavelength.max():.2f} nm")
    print(f"n range: {n.min():.3f} - {n.max():.3f}")
    print(f"κ range: {kappa.min():.6f} - {kappa.max():.6f}")
    
    # Try different fitting methods
    methods = {
        'polynomial': lambda f: f.fit_polynomial(n_degree=6, kappa_degree=5),
    }
    
    if HAS_SCIPY:
        methods['spline'] = lambda f: f.fit_spline(n_s=0.1, kappa_s=0.1)
        methods['interpolation'] = lambda f: f.fit_interpolation(kind='cubic')
    else:
        methods['interpolation'] = lambda f: f.fit_interpolation(kind='cubic')
    
    results = {}
    for method_name, method_func in methods.items():
        fitter = RefractiveIndexFitter(wavelength, n, kappa)
        method_func(fitter)
        errors = fitter.calculate_rms_error()
        results[method_name] = {
            'fitter': fitter,
            'n_rms': errors['n_rms'],
            'kappa_rms': errors['kappa_rms']
        }
        print(f"\n{method_name.upper()} Fit:")
        print(f"  n RMS error: {errors['n_rms']:.6f}")
        print(f"  κ RMS error: {errors['kappa_rms']:.6f}")
    
    # Select best fit (lowest combined RMS)
    best_method = min(results.keys(), 
                      key=lambda k: results[k]['n_rms'] + results[k]['kappa_rms'])
    best_fitter = results[best_method]['fitter']
    
    print(f"\n{'='*80}")
    print(f"Best fit method: {best_method.upper()}")
    print(f"{'='*80}")
    
    # Plot the best fit
    plot_fits(wavelength, n, kappa, best_fitter, 
              save_path='Si_refractive_index_fit.png')
    
    # Example: evaluate at specific wavelengths
    print("\n" + "="*80)
    print("Example Evaluations:")
    print("="*80)
    test_wavelengths = np.array([400, 500, 600, 700, 800, 900])
    n_vals = best_fitter.evaluate_n(test_wavelengths)
    kappa_vals = best_fitter.evaluate_kappa(test_wavelengths)
    eps_vals = best_fitter.evaluate_epsilon(test_wavelengths)
    
    print(f"\n{'Wavelength (nm)':<15} {'n':<10} {'κ':<12} {'ε_real':<12} {'ε_imag':<12}")
    print("-" * 80)
    for wl, n_val, k_val, eps_val in zip(test_wavelengths, n_vals, kappa_vals, eps_vals):
        print(f"{wl:<15.1f} {n_val:<10.4f} {k_val:<12.6f} "
              f"{np.real(eps_val):<12.4f} {np.imag(eps_val):<12.4f}")
    
    return best_fitter


if __name__ == '__main__':
    fitter = main()

