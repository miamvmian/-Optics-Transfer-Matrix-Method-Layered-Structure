"""
Transfer Matrix Method for Multi-Layer Optical Structures

Computes 2x2 transfer matrices for single interfaces and multi-layer structures
across multiple wavelengths. Supports both s-polarization (TE) and p-polarization (TM).
"""

import warnings
from dataclasses import dataclass

import numpy as np

# ============================================================================
# Validation Functions
# ============================================================================


def validate_wavelengths(wavelengths: np.ndarray) -> None:
    """Validate wavelength array."""
    wavelengths = np.asarray(wavelengths)
    if wavelengths.ndim != 1:
        raise ValueError(f"Wavelengths must be 1D array, got {wavelengths.ndim}D")
    if wavelengths.size == 0:
        raise ValueError("Wavelengths array must not be empty")
    if np.any(wavelengths <= 0):
        raise ValueError(
            f"All wavelengths must be positive, got min={np.min(wavelengths):.2e}"
        )
    if np.any(np.isnan(wavelengths)) or np.any(np.isinf(wavelengths)):
        raise ValueError("Wavelengths must not contain NaN or inf values")


def validate_incident_angle(angle_degrees: float) -> None:
    """Validate incident angle."""
    if np.isclose(angle_degrees, 90.0):
        raise ValueError("Incident angle of exactly 90° is not supported")
    if angle_degrees < 0 or angle_degrees >= 90:
        warnings.warn(
            f"Incident angle should be in range [0, 90) degrees; got {angle_degrees}°",
            stacklevel=3,
        )


def validate_permittivity(
    eps: float | complex | np.ndarray,
    wavelengths: np.ndarray | None = None,
    param_name: str = "eps",
) -> None:
    """Validate permittivity array."""
    eps_array = np.asarray(eps, dtype=complex)
    if eps_array.ndim > 1:
        raise ValueError(f"{param_name} must be scalar or 1D array")
    if eps_array.size == 0:
        raise ValueError(f"{param_name} must not be empty")
    if np.any(np.real(eps_array) <= 0):
        raise ValueError(f"{param_name} must have positive real part")
    if np.any(np.isnan(eps_array)) or np.any(np.isinf(eps_array)):
        raise ValueError(f"{param_name} must not contain NaN or inf values")
    if wavelengths is not None and eps_array.size > 1:
        if eps_array.size != wavelengths.size:
            raise ValueError(
                f"{param_name} length ({eps_array.size}) must match wavelengths length ({wavelengths.size})"
            )


def validate_polarization(polarization: str) -> None:
    """Validate polarization mode."""
    if polarization.lower() not in ["s", "p"]:
        raise ValueError(
            f"Polarization must be 's' (TE) or 'p' (TM), got '{polarization}'"
        )


# ============================================================================
# Core Data Structures
# ============================================================================


@dataclass
class WaveField:
    """
    Electromagnetic wave field with forward and backward propagating components.

    Parameters
    ----------
    k_vectors : array_like, shape (N, 2)
        Wave vector array: column 0 is kx (tangential), column 1 is kz (normal)
    amplitude : array_like, shape (N, 2)
        Complex field array: column 0 is forward, column 1 is backward
    """

    k_vectors: np.ndarray
    amplitude: np.ndarray

    def __post_init__(self):
        """Validate and convert inputs."""
        self.k_vectors = np.asarray(self.k_vectors)
        self.amplitude = np.asarray(self.amplitude, dtype=complex)

        if self.k_vectors.ndim != 2 or self.k_vectors.shape[1] != 2:
            raise ValueError(
                f"k_vectors must be 2D array with 2 columns, got {self.k_vectors.shape}"
            )
        if self.amplitude.ndim != 2 or self.amplitude.shape[1] != 2:
            raise ValueError(
                f"amplitude must be 2D array with 2 columns, got {self.amplitude.shape}"
            )
        if self.k_vectors.shape[0] != self.amplitude.shape[0]:
            raise ValueError("k_vectors and amplitude must have same number of rows")

    @property
    def amp_forward(self) -> np.ndarray:
        """Forward propagating amplitude, shape (N,)."""
        return self.amplitude[:, 0]

    @property
    def amp_backward(self) -> np.ndarray:
        """Backward propagating amplitude, shape (N,)."""
        return self.amplitude[:, 1]

    @property
    def n_points(self) -> int:
        """Number of wave vector points."""
        return self.k_vectors.shape[0]

    def _compute_field(self, x: np.ndarray, z: np.ndarray, sign: int) -> np.ndarray:
        """Helper method for field calculations."""
        kx, kz = self.k_vectors[:, 0], self.k_vectors[:, 1]
        is_scalar_x, is_scalar_z = x.ndim == 0, z.ndim == 0

        if is_scalar_x and is_scalar_z:
            phase = kx * x + sign * kz * z
            amp = self.amp_forward if sign > 0 else self.amp_backward
            return amp * np.exp(1j * phase)

        if is_scalar_x:
            x = np.atleast_1d(x)
        if is_scalar_z:
            z = np.atleast_1d(z)

        if x.ndim == 1 and z.ndim == 1:
            phase = kx[:, None] * x[None, :] + sign * kz[:, None] * z[None, :]
            amp = self.amp_forward[:, None] if sign > 0 else self.amp_backward[:, None]
            result = amp * np.exp(1j * phase)
            return result.squeeze() if (is_scalar_x and is_scalar_z) else result
        else:
            phase = (
                kx[:, None, None] * x[None, ...]
                + sign * kz[:, None, None] * z[None, ...]
            )
            amp = (
                self.amp_forward[:, None, None]
                if sign > 0
                else self.amp_backward[:, None, None]
            )
            return amp * np.exp(1j * phase)

    def field_forward(self, x: float | np.ndarray, z: float | np.ndarray) -> np.ndarray:
        """Calculate forward propagating field: E(x,z) = amp_forward * exp(1j * (kx*x + kz*z))."""
        return self._compute_field(np.asarray(x), np.asarray(z), sign=1)

    def field_backward(
        self, x: float | np.ndarray, z: float | np.ndarray
    ) -> np.ndarray:
        """Calculate backward propagating field: E(x,z) = amp_backward * exp(1j * (kx*x - kz*z))."""
        return self._compute_field(np.asarray(x), np.asarray(z), sign=-1)


@dataclass
class Layer:
    """
    Individual layer in a multi-layer photonic structure.

    Parameters
    ----------
    thickness : float
        Layer thickness in meters (must be >= 0)
    optical_property : dict
        Dictionary with 'type' ('permittivity' or 'refractive_index') and 'value'
        The value can be a scalar (constant across wavelengths) or a 1D array
        providing wavelength-dependent data.
    """

    thickness: float
    optical_property: dict

    def __post_init__(self):
        """Validate layer properties."""
        if self.thickness < 0:
            raise ValueError(
                f"Layer thickness must be non-negative, got {self.thickness}"
            )

        if not isinstance(self.optical_property, dict):
            raise ValueError("optical_property must be a dictionary")

        prop_type = self.optical_property.get("type")
        value = self.optical_property.get("value")

        if prop_type not in ["permittivity", "refractive_index"]:
            raise ValueError(
                f"optical_property['type'] must be 'permittivity' or 'refractive_index'"
            )

        # Convert value to numpy array for validation; allow scalars and 1D arrays
        value_array = np.asarray(value, dtype=complex)
        if value_array.ndim > 1:
            raise ValueError("optical_property['value'] must be a scalar or 1D array")
        if value_array.size == 0:
            raise ValueError("optical_property['value'] must not be empty")
        if np.any(np.real(value_array) <= 0):
            raise ValueError(
                f"{prop_type} must have positive real part, got {value_array}"
            )

        # Store the normalized value back (scalar if 0-D, otherwise 1-D array)
        if value_array.ndim == 0:
            normalized_value = value_array.item()
        else:
            normalized_value = value_array
        self.optical_property["value"] = normalized_value

    @property
    def permittivity(self) -> complex | np.ndarray:
        """
        Get permittivity (converted from refractive index if needed).

        Returns
        -------
        complex | np.ndarray
            Scalar value for constant permittivity or 1D array for
            wavelength-dependent permittivity.
        """
        value = np.asarray(self.optical_property["value"], dtype=complex)
        if self.optical_property["type"] == "permittivity":
            eps = value
        else:
            eps = value * value  # ε = n²
        if eps.ndim == 0:
            return eps.item()
        return eps


# ============================================================================
# Wave Vector Calculations
# ============================================================================


def calculate_kz(k0: np.ndarray, kx: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """
    Calculate kz component (normal wave vector) with numerical stability.

    The normal component of the wave vector is calculated from the dispersion relation:
    kz² = k0²*eps - kx²

    A small imaginary part is added to ensure numerical stability when kz² is close to zero
    or negative (evanescent waves).

    Parameters
    ----------
    k0 : np.ndarray
        Free-space wave number, shape (n_wavelengths,)
    kx : np.ndarray
        Tangential wave vector component (conserved across interfaces), shape (n_wavelengths,)
    eps : np.ndarray
        Permittivity of the medium, shape (n_wavelengths,)

    Returns
    -------
    np.ndarray
        Normal wave vector component kz, shape (n_wavelengths,)
        May be complex for evanescent waves (total internal reflection)
    """
    kz_squared = k0**2 * eps - kx**2
    return np.sqrt(kz_squared + 1j * 1e-30)


def calculate_wave_vectors(
    wavelengths: np.ndarray, eps: float | complex | np.ndarray, angle_degrees: float
) -> dict[str, np.ndarray]:
    """
    Calculate wave vector components for electromagnetic wave propagation.

    Computes the wave vector components for a plane wave in a medium with permittivity eps
    at a given incident angle. The wave vector is decomposed into:
    - k0: Free-space wave number (2π/λ)
    - kx: Tangential component (conserved across interfaces via Snell's law)
    - kz: Normal component (calculated from dispersion relation)

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelengths in vacuum, shape (n_wavelengths,)
    eps : float | complex | np.ndarray
        Permittivity of the medium. Can be scalar or array matching wavelengths shape.
    angle_degrees : float
        Incident angle in degrees, range [0, 90)

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        - 'k0': Free-space wave number, shape (n_wavelengths,)
        - 'kx': Tangential wave vector component, shape (n_wavelengths,)
        - 'kz': Normal wave vector component, shape (n_wavelengths,)

    Raises
    ------
    ValueError
        If wavelengths are not positive or angle is out of range
    """
    lda = np.asarray(wavelengths)
    if np.any(lda <= 0):
        raise ValueError("All wavelengths must be positive")
    if angle_degrees < 0 or angle_degrees >= 90:
        raise ValueError(f"Angle must be in range [0, 90) degrees, got {angle_degrees}")

    theta = angle_degrees * np.pi / 180.0
    eps = np.broadcast_to(eps, lda.shape)

    # Free-space wave number: k0 = 2π/λ
    k0 = 2 * np.pi / lda
    # Tangential component: kx = k0 * n * sin(θ) = k0 * sqrt(eps) * sin(θ)
    # This is conserved across interfaces (Snell's law)
    kx = k0 * np.sqrt(eps + 0j) * np.sin(theta)
    # Normal component: kz = sqrt(k0²*eps - kx²)
    kz = calculate_kz(k0, kx, eps)

    return {"k0": k0, "kx": kx, "kz": kz}


# ============================================================================
# Matrix Calculation Classes
# ============================================================================


class SingleInterfaceTMatrix:
    """
    Single interface transfer matrix calculation with wavelength array support.

    Calculates the 2x2 transfer matrix for a single interface between two media.
    The transfer matrix relates field amplitudes on one side of the interface
    to those on the other side.
    """

    def __init__(
        self,
        lda: float | np.ndarray,
        theta: float,
        polarization: str,
        eps_in: float | np.ndarray,
        eps_out: float | np.ndarray,
    ):
        """
        Initialize single interface transfer matrix calculator.

        Parameters
        ----------
        lda : float | np.ndarray
            Wavelength(s) in vacuum, can be scalar or 1D array
        theta : float
            Incident angle in degrees, range [0, 90)
        polarization : str
            Polarization mode: 's' (TE) or 'p' (TM)
        eps_in : float | np.ndarray
            Permittivity of incident medium
        eps_out : float | np.ndarray
            Permittivity of exit medium
        """
        lda_array = np.asarray(lda)
        if lda_array.ndim > 1:
            raise ValueError("Wavelength array must be 1D")
        if np.any(lda_array <= 0):
            raise ValueError("All wavelengths must be positive")
        if np.isclose(theta, 90.0):
            raise ValueError("Incident angle of exactly 90° is not supported")
        if polarization not in ["s", "p"]:
            raise ValueError("Polarization must be 's' or 'p'")

        self.lda = lda_array.reshape(-1) if lda_array.ndim == 0 else lda_array
        self.theta = theta * np.pi / 180
        self.polarization = polarization.lower()
        N = len(self.lda)
        self.eps_in = np.broadcast_to(eps_in, N)
        self.eps_out = np.broadcast_to(eps_out, N)

        self.k0 = 2 * np.pi / self.lda
        self.kx = self.k0 * np.sqrt(self.eps_in + 1j * 0) * np.sin(self.theta)
        self.kz_in = calculate_kz(self.k0, self.kx, self.eps_in)
        self.kz_out = calculate_kz(self.k0, self.kx, self.eps_out)

    def _half_matrix_s(
        self, eps: np.ndarray, kz: np.ndarray, inv: bool = False
    ) -> np.ndarray:
        """
        Half transfer matrix for s-polarization (TE) using cos(theta).

        For s-polarization, the electric field is perpendicular to the plane of incidence.
        The matrix structure is: F = [[1, 1], [-a, a]] where a = sqrt(eps)*cos(theta) = kz/k0

        Parameters
        ----------
        eps : np.ndarray
            Permittivity of the medium, shape (n_wavelengths,)
        kz : np.ndarray
            Normal wave vector component, shape (n_wavelengths,)
        inv : bool, optional
            If True, return the inverse matrix, by default False

        Returns
        -------
        np.ndarray
            Half transfer matrix, shape (n_wavelengths, 2, 2)
        """
        N = len(eps)
        M = np.zeros((N, 2, 2), dtype=complex)
        sqrt_eps = np.sqrt(eps + 0j)
        # cos(theta) = kz / (k0 * sqrt(eps))
        cos_theta = kz / (self.k0 * sqrt_eps)

        if inv:
            # Inverse matrix: F_inv = (1/(2a)) * [[a, -1], [a, 1]]
            # where a = sqrt(eps)*cos(theta) = kz/k0
            a = sqrt_eps * cos_theta
            M[:, 0, 0] = 0.5
            M[:, 0, 1] = -0.5 / a
            M[:, 1, 0] = 0.5
            M[:, 1, 1] = 0.5 / a
        else:
            # Forward matrix: F = [[1, 1], [-a, a]]
            # where a = sqrt(eps)*cos(theta) = kz/k0
            a = sqrt_eps * cos_theta
            M[:, 0, 0] = M[:, 0, 1] = 1.0
            M[:, 1, 0] = -a
            M[:, 1, 1] = a
        return M

    def _half_matrix_p(
        self, eps: np.ndarray, kz: np.ndarray, inv: bool = False
    ) -> np.ndarray:
        """
        Half transfer matrix for p-polarization (TM) using cos(theta).

        For p-polarization, the magnetic field is perpendicular to the plane of incidence.
        The matrix structure is: F = [[b, -b], [1, 1]] where b = cos(theta)/sqrt(eps) = kz/(k0*eps)

        Parameters
        ----------
        eps : np.ndarray
            Permittivity of the medium, shape (n_wavelengths,)
        kz : np.ndarray
            Normal wave vector component, shape (n_wavelengths,)
        inv : bool, optional
            If True, return the inverse matrix, by default False

        Returns
        -------
        np.ndarray
            Half transfer matrix, shape (n_wavelengths, 2, 2)
        """
        N = len(eps)
        M = np.zeros((N, 2, 2), dtype=complex)
        sqrt_eps = np.sqrt(eps + 0j)
        # cos(theta) = kz / (k0 * sqrt(eps))
        cos_theta = kz / (self.k0 * sqrt_eps)

        if inv:
            # Inverse matrix: F_inv = (1/(2b)) * [[1, b], [-1, b]]
            # where b = cos(theta)/sqrt(eps) = kz/(k0*eps)
            b = cos_theta / sqrt_eps
            M[:, 0, 0] = 0.5 * sqrt_eps / cos_theta  # = 0.5 / b
            M[:, 0, 1] = M[:, 1, 1] = 0.5
            M[:, 1, 0] = -0.5 * sqrt_eps / cos_theta  # = -0.5 / b
        else:
            # Forward matrix: F = [[b, -b], [1, 1]]
            # where b = cos(theta)/sqrt(eps) = kz/(k0*eps)
            b = cos_theta / sqrt_eps
            M[:, 0, 0] = b
            M[:, 0, 1] = -b
            M[:, 1, 0] = M[:, 1, 1] = 1.0
        return M

    def full_transfer_matrix(self) -> np.ndarray:
        """
        Calculate transfer matrices: T = F_out_inv @ F_in.

        Constructs the full 2x2 transfer matrix for a single interface by multiplying
        the forward matrix for the incident medium with the inverse matrix for the
        exit medium.

        Returns
        -------
        np.ndarray
            Full transfer matrix, shape (n_wavelengths, 2, 2)
            Relates field amplitudes across the interface
        """
        if self.polarization == "s":
            F_in = self._half_matrix_s(self.eps_in, self.kz_in)
            F_out_inv = self._half_matrix_s(self.eps_out, self.kz_out, inv=True)
        else:
            F_in = self._half_matrix_p(self.eps_in, self.kz_in)
            F_out_inv = self._half_matrix_p(self.eps_out, self.kz_out, inv=True)
        return np.matmul(F_out_inv, F_in)

    @property
    def k_vectors(self) -> dict[str, np.ndarray]:
        """
        Get all calculated wave vectors.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys:
            - 'k0': Free-space wave number, shape (n_wavelengths,)
            - 'kx': Tangential wave vector component, shape (n_wavelengths,)
            - 'kz_in': Normal wave vector component in incident medium, shape (n_wavelengths,)
            - 'kz_out': Normal wave vector component in exit medium, shape (n_wavelengths,)
        """
        return {
            "k0": self.k0,
            "kx": self.kx,
            "kz_in": self.kz_in,
            "kz_out": self.kz_out,
        }

    def __repr__(self) -> str:
        return (
            f"SingleInterfaceTMatrix(lda={len(self.lda)} wavelengths, "
            f"theta={self.theta * 180 / np.pi:.1f}°, polarization='{self.polarization}')"
        )


class LayerPropagationMatrix:
    """
    Layer propagation matrix calculation with wavelength array support.

    Calculates the propagation matrix that describes how wave amplitudes change
    as they propagate through a homogeneous layer of finite thickness.
    """

    def __init__(
        self, lda: float | np.ndarray, theta: float, d: float, eps: float | np.ndarray
    ):
        """
        Initialize layer propagation matrix calculator.

        Parameters
        ----------
        lda : float | np.ndarray
            Wavelength(s) in vacuum, can be scalar or 1D array
        theta : float
            Incident angle in degrees, range [0, 90)
        d : float
            Layer thickness in meters (must be >= 0)
        eps : float | np.ndarray
            Permittivity of the layer
        """
        lda_array = np.asarray(lda)
        if lda_array.ndim > 1:
            raise ValueError("Wavelength array must be 1D")
        if np.any(lda_array <= 0):
            raise ValueError("All wavelengths must be positive")
        if d < 0:
            raise ValueError("Layer thickness must be non-negative")

        self.lda = lda_array.reshape(-1) if lda_array.ndim == 0 else lda_array
        self.theta = theta * np.pi / 180
        self.d = d
        N = len(self.lda)
        self.eps = np.broadcast_to(eps, N)

        self.k0 = 2 * np.pi / self.lda
        self.kx = self.k0 * np.sqrt(self.eps + 1j * 0) * np.sin(self.theta)
        self.kz = calculate_kz(self.k0, self.kx, self.eps)

    def propagation_matrix(self) -> np.ndarray:
        """
        Calculate propagation matrices: P = [[exp(i*kz*d), 0], [0, exp(-i*kz*d)]].

        The propagation matrix describes phase accumulation as waves travel through
        the layer. Forward waves accumulate phase +kz*d, backward waves -kz*d.

        Returns
        -------
        np.ndarray
            Propagation matrix, shape (n_wavelengths, 2, 2)
            P[:, 0, 0] = exp(i*kz*d) for forward wave
            P[:, 1, 1] = exp(-i*kz*d) for backward wave
        """
        N = len(self.lda)
        P = np.zeros((N, 2, 2), dtype=complex)
        # Forward wave: phase accumulation +kz*d
        P[:, 0, 0] = np.exp(1j * self.kz * self.d)
        # Backward wave: phase accumulation -kz*d
        P[:, 1, 1] = np.exp(-1j * self.kz * self.d)
        return P

    def k_vectors(self) -> dict[str, np.ndarray]:
        """Get all calculated wave vectors."""
        return {"k0": self.k0, "kx": self.kx, "kz": self.kz}

    def __repr__(self) -> str:
        return (
            f"LayerPropagationMatrix(lda={len(self.lda)} wavelengths, "
            f"theta={self.theta * 180 / np.pi:.1f}°, d={self.d * 1e9:.1f}nm)"
        )


class Interface:
    """
    Interface between two media with Fresnel equations.

    Calculates Fresnel reflection and transmission coefficients for an interface
    between two media with different permittivities.
    """

    def __init__(
        self,
        eps1: complex | np.ndarray,
        eps2: complex | np.ndarray,
        kx: np.ndarray,
        polarization: str,
    ) -> None:
        """
        Initialize interface.

        Parameters
        ----------
        eps1 : complex | np.ndarray
            Permittivity of first medium (incident side)
        eps2 : complex | np.ndarray
            Permittivity of second medium (transmitted side)
        kx : np.ndarray
            Tangential wave vector component (conserved across interface)
        polarization : str
            Polarization mode: 's' (TE) or 'p' (TM)
        """
        if polarization.lower() not in ["s", "p"]:
            raise ValueError(f"Polarization must be 's' or 'p', got '{polarization}'")

        self.polarization = polarization.lower()
        self.eps1 = np.asarray(eps1, dtype=complex)
        self.eps2 = np.asarray(eps2, dtype=complex)
        self.kx = np.asarray(kx, dtype=complex)

        if np.any(np.real(self.eps1) <= 0) or np.any(np.real(self.eps2) <= 0):
            raise ValueError("Permittivities must have positive real part")

    def kz_components(self, k0: np.ndarray) -> dict[str, np.ndarray]:
        """
        Calculate normal wave vector components for both media.

        Parameters
        ----------
        k0 : np.ndarray
            Free-space wave number, shape (n_wavelengths,)

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys:
            - 'kz1': Normal component in medium 1, shape (n_wavelengths,)
            - 'kz2': Normal component in medium 2, shape (n_wavelengths,)
        """
        k0 = np.asarray(k0)
        kz1 = calculate_kz(k0, self.kx, self.eps1)
        kz2 = calculate_kz(k0, self.kx, self.eps2)
        return {"kz1": kz1, "kz2": kz2}

    def fresnel_coefficients(self, k0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate Fresnel reflection and transmission coefficients.

        Computes the amplitude reflection (r) and transmission (t) coefficients
        using the Fresnel equations for s-polarization (TE) or p-polarization (TM).

        Parameters
        ----------
        k0 : np.ndarray
            Free-space wave number, shape (n_wavelengths,)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (r, t) where:
            - r: Reflection coefficient (amplitude), shape (n_wavelengths,)
            - t: Transmission coefficient (amplitude), shape (n_wavelengths,)

        Notes
        -----
        For s-polarization: r = (kz1 - kz2) / (kz1 + kz2)
        For p-polarization: r = (eps2*kz1 - eps1*kz2) / (eps2*kz1 + eps1*kz2)
        """
        kz_dict = self.kz_components(k0)
        kz1, kz2 = kz_dict["kz1"], kz_dict["kz2"]

        if self.polarization == "s":
            # s-polarization (TE): electric field perpendicular to plane of incidence
            r = (kz1 - kz2) / (kz1 + kz2)
            t = 2 * kz1 / (kz1 + kz2)
        else:
            # p-polarization (TM): magnetic field perpendicular to plane of incidence
            r = (self.eps2 * kz1 - self.eps1 * kz2) / (
                self.eps2 * kz1 + self.eps1 * kz2
            )
            t = 2 * self.eps1 * kz1 / (self.eps2 * kz1 + self.eps1 * kz2)
        return r, t


# ============================================================================
# Multi-Layer Structure
# ============================================================================


class MultiLayerStructure:
    """
    Multi-layer photonic structure for transfer matrix calculations.

    This class implements the Transfer Matrix Method (TMM) for calculating
    optical properties (reflectance, transmittance) of multi-layer structures.
    Supports both s-polarization (TE) and p-polarization (TM) at arbitrary
    incident angles.

    The structure is defined as:
    Incident Medium | Layer 1 | Layer 2 | ... | Layer N | Exit Medium

    Attributes
    ----------
    wavelengths : np.ndarray
        Wavelengths in vacuum, shape (n_wavelengths,)
    angle_degrees : float
        Incident angle in degrees
    polarization : str
        Polarization mode: 's' (TE) or 'p' (TM)
    layers : list[Layer]
        List of Layer objects defining the structure
    eps_incident : np.ndarray
        Permittivity of incident medium, shape (n_wavelengths,)
    eps_exit : np.ndarray
        Permittivity of exit medium, shape (n_wavelengths,)
    """

    def __init__(
        self,
        wavelengths: float | np.ndarray,
        angle_degrees: float,
        polarization: str,
        layers: list[Layer],
        eps_incident: float | complex | np.ndarray,
        eps_exit: float | complex | np.ndarray,
    ):
        """
        Initialize multi-layer structure.

        Parameters
        ----------
        wavelengths : float | np.ndarray
            Wavelength(s) in vacuum, can be scalar or 1D array
        angle_degrees : float
            Incident angle in degrees, range [0, 90)
        polarization : str
            Polarization mode: 's' (TE) or 'p' (TM)
        layers : list[Layer]
            List of Layer objects, ordered from incident to exit side
        eps_incident : float | complex | np.ndarray
            Permittivity of incident medium
        eps_exit : float | complex | np.ndarray
            Permittivity of exit medium

        Raises
        ------
        ValueError
            If input validation fails (invalid wavelengths, angle, polarization, etc.)
        """
        wavelengths = np.asarray(wavelengths, dtype=float)
        validate_wavelengths(wavelengths)
        validate_incident_angle(angle_degrees)
        validate_polarization(polarization)
        if not isinstance(layers, list):
            raise ValueError(f"layers must be a list, got {type(layers)}")
        for i, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise ValueError(f"layers[{i}] must be a Layer instance")
        validate_permittivity(eps_incident, wavelengths, "eps_incident")
        validate_permittivity(eps_exit, wavelengths, "eps_exit")

        self.wavelengths = wavelengths
        self.n_wavelengths = len(wavelengths)
        self.angle_degrees = angle_degrees
        self.angle_rad = angle_degrees * np.pi / 180.0
        self.polarization = polarization.lower()
        self.layers = layers
        self.n_layers = len(layers)
        self.eps_incident = np.broadcast_to(eps_incident, self.n_wavelengths)
        self.eps_exit = np.broadcast_to(eps_exit, self.n_wavelengths)
        # Normalize layer permittivities (allow scalar or wavelength-dependent arrays)
        self._layer_permittivities = [
            self._normalize_permittivity(layer.permittivity, layer_index=i)
            for i, layer in enumerate(self.layers)
        ]

        self.k0 = 2 * np.pi / self.wavelengths
        self.kx = self.k0 * np.sqrt(self.eps_incident + 0j) * np.sin(self.angle_rad)

    def _normalize_permittivity(
        self, eps: complex | float | np.ndarray, layer_index: int | None = None
    ) -> np.ndarray:
        """
        Normalize permittivity to array of shape (n_wavelengths,).

        Parameters
        ----------
        eps : complex | float | np.ndarray
            Permittivity value(s) to normalize.
        layer_index : int | None, optional
            Index of the layer for error reporting. If None, refers to media.

        Returns
        -------
        np.ndarray
            Array of permittivities for each wavelength.
        """

        eps_array = np.asarray(eps, dtype=complex)
        if eps_array.ndim == 0:
            return np.broadcast_to(eps_array, self.n_wavelengths)
        if eps_array.ndim == 1:
            if len(eps_array) != self.n_wavelengths:
                prefix = f"Layer {layer_index} " if layer_index is not None else ""
                raise ValueError(
                    f"{prefix}permittivity array length ({len(eps_array)}) "
                    f"must match number of wavelengths ({self.n_wavelengths})"
                )
            return eps_array
        raise ValueError(
            "Permittivity values must be scalar or 1D array matching wavelengths"
        )

    def _calculate_layer_kz(self, eps: np.ndarray) -> np.ndarray:
        """
        Calculate kz component (normal wave vector) with numerical stability.
        The normal component of the wave vector is calculated from the dispersion relation:
        kz² = k0²*eps - kx²
        A small imaginary part is added to ensure numerical stability when kz² is close to zero
        or negative (evanescent waves).

        Parameters
        ----------
        eps : np.ndarray
            Permittivity of the medium, shape (n_wavelengths,)

        Returns
        -------
        np.ndarray
            Normal wave vector component kz, shape (n_wavelengths,),
            may be complex for evanescent waves (total internal reflection)
        """
        kz_squared = self.k0**2 * eps - self.kx**2
        # Add small imaginary part for numerical stability
        return np.sqrt(kz_squared + 1j * 1e-30)

    def _create_interface_half_matrices(self, eps) -> tuple[np.ndarray, np.ndarray]:
        """
        Create half transfer matrices for a single medium using cos(theta).

        This method creates the forward matrix F_in and its inverse F_out_inv
        for a medium with permittivity eps. These matrices are used to construct
        interface transfer matrices between different media.

        Parameters
        ----------
        eps : np.ndarray
            Permittivity of the medium, shape (n_wavelengths,)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (F_in, F_out_inv) where:
            - F_in: Forward matrix for the medium, shape (n_wavelengths, 2, 2)
            - F_out_inv: Inverse of F_in, shape (n_wavelengths, 2, 2)

        Notes
        -----
        The matrices are constructed using cos(theta) = kz / (k0 * sqrt(eps))
        for numerical stability and physical correctness.
        """
        kz = self._calculate_layer_kz(eps)
        N = self.n_wavelengths
        F_in = np.zeros((N, 2, 2), dtype=complex)
        F_out_inv = np.zeros((N, 2, 2), dtype=complex)

        # Calculate cos(theta) for the medium: cos(theta) = kz / (k0 * sqrt(eps))
        sqrt_eps = np.sqrt(eps + 0j)
        cos_theta = kz / (self.k0 * sqrt_eps)

        if self.polarization == "s":
            # s-polarization: F_in = [[1, 1], [-sqrt(eps)*cos(theta), sqrt(eps)*cos(theta)]]
            # For F = [[1, 1], [-a, a]], the inverse is F_inv = (1/(2a)) * [[a, -1], [a, 1]]
            # where a = sqrt(eps)*cos(theta) = kz/k0
            kz_over_k0 = sqrt_eps * cos_theta  # This equals kz/k0
            F_in[:, 0, :] = 1.0
            F_in[:, 1, 0] = -kz_over_k0
            F_in[:, 1, 1] = kz_over_k0
            # F_out_inv is the inverse of F_in
            # F_out_inv = [[0.5, -1/(2a)], [0.5, 1/(2a)]]
            F_out_inv[:, 0, 0] = 0.5
            F_out_inv[:, 0, 1] = -0.5 / kz_over_k0
            F_out_inv[:, 1, 0] = 0.5
            F_out_inv[:, 1, 1] = 0.5 / kz_over_k0
        else:
            # p-polarization: F_in = [[cos(theta)/sqrt(eps), -cos(theta)/sqrt(eps)], [1, 1]]
            # For F = [[b, -b], [1, 1]] where b = cos(theta)/sqrt(eps) = kz/(k0*eps),
            # the inverse is F_inv = (1/(2b)) * [[1, b], [-1, b]]
            kz_over_k0_eps = cos_theta / sqrt_eps  # This equals kz/(k0*eps)
            F_in[:, 0, 0] = kz_over_k0_eps
            F_in[:, 0, 1] = -kz_over_k0_eps
            F_in[:, 1, :] = 1.0
            # F_out_inv is the inverse of F_in
            # F_out_inv = [[1/(2b), 0.5], [-1/(2b), 0.5]]
            F_out_inv[:, 0, 0] = 0.5 / kz_over_k0_eps
            F_out_inv[:, 0, 1] = 0.5
            F_out_inv[:, 1, 0] = -0.5 / kz_over_k0_eps
            F_out_inv[:, 1, 1] = 0.5

        return F_in, F_out_inv

    def _create_propagation_matrix(self, layer_index: int) -> np.ndarray:
        """
        Create propagation matrix for a layer.

        The propagation matrix describes how the forward and backward wave amplitudes
        change as the wave propagates through a layer of thickness d:
        P = [[exp(i*kz*d), 0], [0, exp(-i*kz*d)]]

        The forward wave (index 0) accumulates phase +kz*d, while the backward wave
        (index 1) accumulates phase -kz*d.

        Parameters
        ----------
        layer_index : int
            Index of the layer whose propagation matrix is computed.

        Returns
        -------
        np.ndarray
            Propagation matrix, shape (n_wavelengths, 2, 2)
            P[:, 0, 0] = exp(i*kz*d) for forward wave
            P[:, 1, 1] = exp(-i*kz*d) for backward wave
        """
        eps = self._layer_permittivities[layer_index]
        kz = self._calculate_layer_kz(eps)
        N = self.n_wavelengths
        P = np.zeros((N, 2, 2), dtype=complex)
        # Forward wave: accumulates phase +kz*d
        thickness = self.layers[layer_index].thickness
        P[:, 0, 0] = np.exp(1j * kz * thickness)
        # Backward wave: accumulates phase -kz*d
        P[:, 1, 1] = np.exp(-1j * kz * thickness)
        return P

    def total_transfer_matrix(self) -> np.ndarray:
        """
        Calculate the total transfer matrix for the multi-layer structure.

        The total transfer matrix relates the field amplitudes in the incident medium
        to those in the exit medium: [A_N^+, A_N^-] = T @ [A_1^+, A_1^-]

        For zero layers (single interface), uses SingleInterfaceTMatrix directly.
        For one or more layers, constructs the matrix by multiplying interface and
        propagation matrices in the correct order.

        Returns
        -------
        np.ndarray
            Total transfer matrix, shape (n_wavelengths, 2, 2)
            T[:, 0, 0] relates A_1^+ to A_N^+
            T[:, 0, 1] relates A_1^- to A_N^+
            T[:, 1, 0] relates A_1^+ to A_N^-
            T[:, 1, 1] relates A_1^- to A_N^-
        """
        # Zero layers: single interface case
        if self.n_layers == 0:
            tm = SingleInterfaceTMatrix(
                lda=self.wavelengths,
                theta=self.angle_degrees,
                polarization=self.polarization,
                eps_in=self.eps_incident,
                eps_out=self.eps_exit,
            )
            return tm.full_transfer_matrix()

        # One or more layers: construct matrices for each interface and layer
        # F_in: forward matrix for incident medium
        # F_0_inv: inverse matrix for first layer (used to cancel F_in at first interface)
        F_in, F_0_inv = self._create_interface_half_matrices(self.eps_incident)
        inner_interface_matrices = []  # Forward matrices for each layer
        inner_interface_matrices_inv = [
            F_0_inv
        ]  # Inverse matrices (starts with F_0_inv)

        # Create interface matrices for transitions between layers
        # For N layers, we need N-1 inter-layer interfaces
        for i in range(self.n_layers - 1):
            # Matrix for layer i (forward direction)
            eps_layer = self._layer_permittivities[i]
            F_i, F_i_inv = self._create_interface_half_matrices(eps_layer)
            inner_interface_matrices.append(F_i)
            inner_interface_matrices_inv.append(F_i_inv)

        # F_last: forward matrix for exit medium (last layer to exit)
        # F_out_inv: inverse matrix for exit medium
        F_last, F_out_inv = self._create_interface_half_matrices(self.eps_exit)
        inner_interface_matrices.append(F_last)

        # Create propagation matrices for each layer
        propagation_matrices = [
            self._create_propagation_matrix(idx) for idx in range(self.n_layers)
        ]

        # Multiply all matrices in the correct order to get total transfer matrix
        return multiply_transfer_matrices(
            F_in,
            F_out_inv,
            inner_interface_matrices,
            propagation_matrices,
            inner_interface_matrices_inv,
        )

    def wave_vectors(self) -> dict[str, np.ndarray]:
        """
        Get all wave vector components for the structure.

        Returns wave vector components for the incident medium, exit medium,
        and all intermediate layers. The kx component is conserved across
        all interfaces (Snell's law), while kz varies with permittivity.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys:
            - 'k0': Free-space wave number, shape (n_wavelengths,)
            - 'kx': Tangential wave vector (conserved), shape (n_wavelengths,)
            - 'kz_incident': Normal component in incident medium, shape (n_wavelengths,)
            - 'kz_exit': Normal component in exit medium, shape (n_wavelengths,)
            - 'kz_layers': List of normal components for each layer,
                          each element shape (n_wavelengths,)
        """
        kz_incident = self._calculate_layer_kz(self.eps_incident)
        kz_exit = self._calculate_layer_kz(self.eps_exit)
        kz_layers = [
            self._calculate_layer_kz(self._layer_permittivities[i])
            for i in range(self.n_layers)
        ]
        return {
            "k0": self.k0,
            "kx": self.kx,
            "kz_incident": kz_incident,
            "kz_exit": kz_exit,
            "kz_layers": kz_layers,
        }

    def reflectance(self) -> np.ndarray:
        """
        Calculate reflectance: R = |r|².

        According to the transfer matrix method:
        r = A_1^- / A_1^+ = -T_21 / T_22
        where T_21 = t_matrix[:, 1, 0] and T_22 = t_matrix[:, 1, 1]
        """
        t_matrix = self.total_transfer_matrix()
        r = -t_matrix[:, 1, 0] / t_matrix[:, 1, 1]
        return np.abs(r) ** 2

    def transmittance(self) -> np.ndarray:
        """
        Calculate transmittance with proper power flow correction.

        The calculation method depends on the number of layers:

        **Zero layers (single interface):**
        Uses full power flow correction to account for different incident/exit media:
        - s-polarization: t_s = T_11 - (T_21 / T_22) * T_12
          T_s = Re((sqrt(eps_N) * cos(theta_N))*) / (sqrt(eps_1) * cos(theta_1)) * |t_s|²
        - p-polarization: t_p = (sqrt(eps_1) / sqrt(eps_N)) * (T_11 - (T_21 / T_22) * T_12)
          T_p = Re((sqrt(eps_N) * cos(theta_N))*) / (sqrt(eps_1) * cos(theta_1)) * |t_p|²

        **One or more layers:**
        Uses simplified formula (power flow correction already accounted for in matrix construction):
        - s-polarization: t_s = T_11 - (T_21 / T_22) * T_12, T_s = |t_s|²
        - p-polarization: t_p = T_11 - (T_21 / T_22) * T_12, T_p = |t_p|²

        where sqrt(eps_n) * cos(theta_n) = kz_n / k_0

        Returns
        -------
        np.ndarray
            Transmittance values, shape (n_wavelengths,), in range [0, inf)
        """
        t_matrix = self.total_transfer_matrix()

        # Common term: t_base = T_11 - (T_21 / T_22) * T_12
        t_base = (
            t_matrix[:, 0, 0]
            - (t_matrix[:, 1, 0] / t_matrix[:, 1, 1]) * t_matrix[:, 0, 1]
        )

        # Calculate kz components
        kz_incident = self._calculate_layer_kz(self.eps_incident)
        kz_exit = self._calculate_layer_kz(self.eps_exit)

        # sqrt(eps_n) * cos(theta_n) = kz_n / k_0

        # Handle zero layer case separately (single interface)
        if self.n_layers == 0:
            sqrt_eps_incident = np.sqrt(self.eps_incident)
            sqrt_eps_exit = np.sqrt(self.eps_exit)
            sqrt_eps_cos_theta_incident = kz_incident / self.k0
            sqrt_eps_cos_theta_exit = kz_exit / self.k0

            if self.polarization == "s":
                # s-polarization: t_s = T_11 - (T_21 / T_22) * T_12
                t_s = t_base
                # T_s = Re((sqrt(eps_N) * cos(theta_N))*) / (sqrt(eps_1) * cos(theta_1)) * |t_s|²
                T_val = (
                    np.real(np.conj(sqrt_eps_cos_theta_exit))
                    / sqrt_eps_cos_theta_incident
                    * np.abs(t_s) ** 2
                )
            else:
                # p-polarization: t_p = (sqrt(eps_1) / sqrt(eps_N)) * (T_11 - (T_21 / T_22) * T_12)
                t_p = (sqrt_eps_incident / sqrt_eps_exit) * t_base
                # T_p = Re((sqrt(eps_N) * cos(theta_N))*) / (sqrt(eps_1) * cos(theta_1)) * |t_p|²
                T_val = (
                    np.real(np.conj(sqrt_eps_cos_theta_exit))
                    / sqrt_eps_cos_theta_incident
                    * np.abs(t_p) ** 2
                )
        else:
            # Cases with layers (1+ layers): use simplified formula
            if self.polarization == "s":
                # s-polarization: t_s = T_11 - (T_21 / T_22) * T_12
                t_s = t_base
                T_val = np.abs(t_s) ** 2
            else:
                # p-polarization: t_p = T_11 - (T_21 / T_22) * T_12
                t_p = t_base
                T_val = np.abs(t_p) ** 2

        return np.real(T_val)

    def __repr__(self) -> str:
        return (
            f"MultiLayerStructure({self.n_wavelengths} wavelengths, "
            f"{self.n_layers} layer(s), angle={self.angle_degrees}°, "
            f"polarization='{self.polarization}')"
        )


# ============================================================================
# Helper Functions
# ============================================================================


def multiply_transfer_matrices(
    F_in: np.ndarray,
    F_out_inv: np.ndarray,
    inner_interface_matrices: list[np.ndarray],
    propagation_matrices: list[np.ndarray],
    inner_interface_matrices_inv: list[np.ndarray],
) -> np.ndarray:
    """
    Multiply interface and propagation matrices to form total transfer matrix.

    Constructs the total transfer matrix by multiplying interface and propagation
    matrices in the correct order. The multiplication order follows the physical
    propagation path through the structure.

    For N layers, the matrix multiplication order is:
    T = F_out_inv @ F_N @ P_N @ F_N_inv @ ... @ F_1 @ P_1 @ F_1_inv @ F_in

    Where:
    - F_in: Forward matrix for incident medium
    - F_i_inv: Inverse matrix for layer i (cancels interface effects)
    - P_i: Propagation matrix through layer i
    - F_i: Forward matrix for layer i (at next interface)
    - F_out_inv: Inverse matrix for exit medium

    Parameters
    ----------
    F_in : np.ndarray
        Forward matrix for incident medium, shape (n_wavelengths, 2, 2)
    F_out_inv : np.ndarray
        Inverse matrix for exit medium, shape (n_wavelengths, 2, 2)
    inner_interface_matrices : list[np.ndarray]
        List of forward matrices for each layer and exit medium,
        length = n_layers, each shape (n_wavelengths, 2, 2)
    propagation_matrices : list[np.ndarray]
        List of propagation matrices for each layer,
        length = n_layers, each shape (n_wavelengths, 2, 2)
    inner_interface_matrices_inv : list[np.ndarray] | None
        List of inverse matrices for each layer,
        length = n_layers, each shape (n_wavelengths, 2, 2)

    Returns
    -------
    np.ndarray
        Total transfer matrix, shape (n_wavelengths, 2, 2)
    """
    n_layers = len(propagation_matrices)

    # Special case: single layer (optimized path)
    if n_layers == 1:
        F_0_inv = inner_interface_matrices_inv[0]
        F_0 = inner_interface_matrices[0]
        P_0 = propagation_matrices[0]
        # Multiplication order: F_0_inv @ F_in, then P_0, then F_0, then F_out_inv
        t_matrix = np.matmul(F_0_inv, F_in)
        t_matrix = np.matmul(P_0, t_matrix)
        t_matrix = np.matmul(F_0, t_matrix)
        return np.matmul(F_out_inv, t_matrix)

    # General case: multiple layers
    # Start with incident medium matrix
    t_matrix = F_in.copy()
    # For each layer, apply: F_i_inv @ P_i @ F_i
    for i in range(n_layers):
        # Apply inverse matrix (interface from previous medium to layer i)
        t_matrix = np.matmul(inner_interface_matrices_inv[i], t_matrix)
        # Apply propagation through layer i
        t_matrix = np.matmul(propagation_matrices[i], t_matrix)
        # Apply forward matrix (interface from layer i to next medium)
        t_matrix = np.matmul(inner_interface_matrices[i], t_matrix)
    # Final multiplication with exit medium inverse matrix
    return np.matmul(F_out_inv, t_matrix)
