# Single Interface Transfer Matrix Method for Wavelength Arrays
# Computes 2x2 transfer matrices for a single interface across multiple wavelengths

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class WaveField:
    """
    Electromagnetic wave field with forward and backward propagating components.

    Stores complex field data across multiple wave vectors with shape (N, 2)
    where N is the number of wave vector points and the 2 columns represent
    forward and backward propagating field components.

    Parameters
    ----------
    k_vectors : array_like
        Wave vector array, shape (N,)
    field : array_like
        Complex field array, shape (N, 2)
        Column 0: forward propagating component
        Column 1: backward propagating component

    Attributes
    ----------
    k_vectors : np.ndarray
        Wave vector array, shape (N,)
    field : np.ndarray
        Complex field array, shape (N, 2)

    Properties
    ----------
    forward : np.ndarray
        Forward propagating component, shape (N,)
    backward : np.ndarray
        Backward propagating component, shape (N,)
    n_points : int
        Number of wave vector points

    Examples
    --------
    >>> import numpy as np
    >>> k_vectors = np.array([1.0e7, 1.2e7, 1.4e7])  # Wave vectors can be real or complex
    >>> forward_amp = np.array([1.0+0j, 0.9+0.1j, 0.8+0.2j])
    >>> backward_amp = np.array([0.1+0j, 0.15+0.05j, 0.2+0.1j])
    >>> field_data = np.column_stack([forward_amp, backward_amp])
    >>> wave_field = WaveField(k_vectors, field_data)
    >>> print(wave_field.forward)   # Forward propagating
    >>> print(wave_field.backward)  # Backward propagating
    >>> print(wave_field.n_points)  # Number of points
    """  # noqa: E501

    k_vectors: np.ndarray
    field: np.ndarray

    def __post_init__(self):
        """
        Validate and convert inputs after dataclass initialization.

        Raises
        ------
        ValueError
            If inputs don't meet validation requirements
        """
        # Convert inputs to numpy arrays if needed
        self.k_vectors = np.asarray(self.k_vectors)
        self.field = np.asarray(self.field)

        # Validate k_vectors
        if self.k_vectors.ndim != 1:
            raise ValueError(
                f"k_vectors must be 1D array, got shape {self.k_vectors.shape}"
            )

        if len(self.k_vectors) == 0:
            raise ValueError("k_vectors must have at least 1 element")

        # Validate field
        if self.field.ndim != 2:
            raise ValueError(f"field must be 2D array, got shape {self.field.shape}")

        if self.field.shape[1] != 2:
            raise ValueError(f"field must have 2 columns, got {self.field.shape[1]}")

        # Convert field to complex if needed
        if not np.iscomplexobj(self.field):
            self.field = self.field.astype(complex)

    @property
    def forward(self) -> np.ndarray:
        """Forward propagating component, shape (N,)."""
        return self.field[:, 0]

    @property
    def backward(self) -> np.ndarray:
        """Backward propagating component, shape (N,)."""
        return self.field[:, 1]

    @property
    def n_points(self) -> int:
        """Number of wave vector points."""
        return len(self.k_vectors)


class SingleInterfaceTMatrix:
    """
    Single interface transfer matrix calculation with wavelength array support.

    Computes the 2x2 transfer matrix for a single interface at multiple wavelengths
    with a fixed incident angle. Supports both s-polarization (TE) and p-polarization (TM).

    Parameters
    ----------
    lda : array_like
        Wavelength array in meters, shape (N,)
    theta : float
        Incident angle in degrees (single value, not array)
    polarization : str
        Polarization: 's' for TE mode, 'p' for TM mode
    eps_in : float or array_like
        Input medium permittivity (scalar or shape (N,))
    eps_out : float or array_like
        Output medium permittivity (scalar or shape (N,))

    Attributes
    ----------
    lda : np.ndarray
        Wavelength array in meters, shape (N,)
    theta : float
        Incident angle in radians
    polarization : str
        Polarization mode
    eps_in : np.ndarray
        Input medium permittivity, shape (N,)
    eps_out : np.ndarray
        Output medium permittivity, shape (N,)
    k0 : np.ndarray
        Free space wave vectors, shape (N,)
    kx : np.ndarray
        Tangential wave vectors, shape (N,)
    kz_in : np.ndarray
        Input medium z-components, shape (N,)
    kz_out : np.ndarray
        Output medium z-components, shape (N,)

    Examples
    --------
    >>> lda = np.array([500e-9, 600e-9, 700e-9])  # 3 wavelengths
    >>> theta = 30.0  # 30 degrees
    >>> eps_in = 1.0  # Air
    >>> eps_out = 2.25  # Glass (constant)
    >>> tm = SingleInterfaceTMatrix(lda, theta, 's', eps_in, eps_out)
    >>> T_matrices = tm.full_transfer_matrix()  # Shape (3, 2, 2)
    """  # noqa: E501

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
        lda : array_like
            Wavelength array in meters
        theta : float
            Incident angle in degrees
        polarization : str
            Polarization: 's' for TE, 'p' for TM
        eps_in : float or array_like
            Input medium permittivity
        eps_out : float or array_like
            Output medium permittivity
        """
        # Validate and convert inputs
        self._validate_inputs(lda, theta, polarization, eps_in, eps_out)

        # Convert to arrays and ensure proper shapes
        self.lda = np.asarray(lda, dtype=float)
        if self.lda.ndim == 0:
            self.lda = self.lda.reshape(1)

        self.theta = theta * np.pi / 180  # Convert to radians
        self.polarization = polarization

        # Broadcast permittivities to match wavelength array shape
        N = len(self.lda)
        self.eps_in = np.broadcast_to(eps_in, N)
        self.eps_out = np.broadcast_to(eps_out, N)

        # Calculate wave vectors
        self.k0 = 2 * np.pi / self.lda
        self.kx = self.k0 * np.sqrt(self.eps_in + 1j * 0) * np.sin(self.theta)

        # Calculate z-components of wave vectors
        self.kz_in = self._calculate_kz(self.eps_in)
        self.kz_out = self._calculate_kz(self.eps_out)

    def _validate_inputs(self, lda, theta, polarization, eps_in, eps_out):
        """Validate input parameters."""
        # Validate wavelength
        lda_array = np.asarray(lda)
        if lda_array.ndim > 1:
            raise ValueError("Wavelength array must be 1D")
        if np.any(lda_array <= 0):
            raise ValueError("All wavelengths must be positive")

        # Validate angle
        if np.isclose(theta, 90.0):
            raise ValueError(
                "Incident angle of exactly 90° (grazing incidence) "
                "is not supported due to numerical instability"
            )
        if theta < 0 or theta > 90:
            warnings.warn(
                "Incident angle should be in range [0, 90) degrees; "
                "near 90° results may be unstable",
                stacklevel=2,
            )

        # Validate polarization
        if polarization not in ["s", "p"]:
            raise ValueError("Polarization must be 's' or 'p'")

        # Validate permittivities
        for name, eps in [("eps_in", eps_in), ("eps_out", eps_out)]:
            eps_array = np.asarray(eps)
            if eps_array.ndim > 1:
                raise ValueError(f"{name} must be scalar or 1D array")
            if eps_array.size > 1 and eps_array.size != lda_array.size:
                raise ValueError(f"{name} length must match wavelength array length")

    def _calculate_kz(self, eps: np.ndarray) -> np.ndarray:
        """
        Calculate z-component of wave vector with proper branch selection.

        Parameters
        ----------
        eps : np.ndarray
            Permittivity array, shape (N,)

        Returns
        -------
        np.ndarray
            z-component of wave vector, shape (N,)
        """
        kz_squared = self.k0**2 * eps - self.kx**2
        # Add small imaginary part for numerical stability
        kz = np.sqrt(kz_squared + 1j * 1e-30)
        return kz

    def _half_transfer_matrix_s(self, eps: np.ndarray, kz: np.ndarray) -> np.ndarray:
        """
        Half field transfer matrix at interface for s-polarization (TE mode).

        Parameters
        ----------
        eps : np.ndarray
            Permittivity array, shape (N,)
        kz : np.ndarray
            z-component of wave vector, shape (N,)

        Returns
        -------
        np.ndarray
            Half field transfer matrices at interface, shape (N, 2, 2)
        """
        N = len(eps)
        F = np.zeros((N, 2, 2), dtype=complex)

        # F[0,0] = 1, F[0,1] = 1
        F[:, 0, 0] = 1.0
        F[:, 0, 1] = 1.0

        # F[1,0] = -kz/k0, F[1,1] = kz/k0
        F[:, 1, 0] = -kz / self.k0
        F[:, 1, 1] = kz / self.k0

        return F

    def _half_transfer_matrix_s_inv(
        self, eps: np.ndarray, kz: np.ndarray
    ) -> np.ndarray:
        """
        Inverse half field transfer matrix at interface for s-polarization (TE mode).

        Parameters
        ----------
        eps : np.ndarray
            Permittivity array, shape (N,)
        kz : np.ndarray
            z-component of wave vector, shape (N,)

        Returns
        -------
        np.ndarray
            Inverse half field transfer matrices at interface, shape (N, 2, 2)
        """
        N = len(eps)
        F_inv = np.zeros((N, 2, 2), dtype=complex)

        # F_inv[0,0] = 0.5, F_inv[0,1] = -0.5*k0/kz
        F_inv[:, 0, 0] = 0.5
        F_inv[:, 0, 1] = -0.5 * self.k0 / kz

        # F_inv[1,0] = 0.5, F_inv[1,1] = 0.5*k0/kz
        F_inv[:, 1, 0] = 0.5
        F_inv[:, 1, 1] = 0.5 * self.k0 / kz

        return F_inv

    def _half_transfer_matrix_p(self, eps: np.ndarray, kz: np.ndarray) -> np.ndarray:
        """
        Half field transfer matrix at interface for p-polarization (TM mode).

        Parameters
        ----------
        eps : np.ndarray
            Permittivity array, shape (N,)
        kz : np.ndarray
            z-component of wave vector, shape (N,)

        Returns
        -------
        np.ndarray
            Half field transfer matrices at interface, shape (N, 2, 2)
        """
        N = len(eps)
        F = np.zeros((N, 2, 2), dtype=complex)

        # F[0,0] = kz/(k0*eps), F[0,1] = -kz/(k0*eps)
        F[:, 0, 0] = kz / (self.k0 * eps)
        F[:, 0, 1] = -kz / (self.k0 * eps)

        # F[1,0] = 1, F[1,1] = 1
        F[:, 1, 0] = 1.0
        F[:, 1, 1] = 1.0

        return F

    def _half_transfer_matrix_p_inv(
        self, eps: np.ndarray, kz: np.ndarray
    ) -> np.ndarray:
        """
        Inverse half field transfer matrix at interface for p-polarization (TM mode).

        Parameters
        ----------
        eps : np.ndarray
            Permittivity array, shape (N,)
        kz : np.ndarray
            z-component of wave vector, shape (N,)

        Returns
        -------
        np.ndarray
            Inverse half field transfer matrices at interface, shape (N, 2, 2)
        """
        N = len(eps)
        F_inv = np.zeros((N, 2, 2), dtype=complex)

        # F_inv[0,0] = 0.5*k0*eps/kz, F_inv[0,1] = 0.5
        F_inv[:, 0, 0] = 0.5 * (self.k0 * eps) / kz
        F_inv[:, 0, 1] = 0.5

        # F_inv[1,0] = -0.5*k0*eps/kz, F_inv[1,1] = 0.5
        F_inv[:, 1, 0] = -0.5 * (self.k0 * eps) / kz
        F_inv[:, 1, 1] = 0.5

        return F_inv

    def full_transfer_matrix(self) -> np.ndarray:
        """
        Calculate transfer matrices for the single interface at all wavelengths.

        For a single interface (no layer thickness), the transfer matrix is:
        T = F_out * F_in^{-1}

        Returns
        -------
        np.ndarray
            Transfer matrices, shape (N, 2, 2) where N is number of wavelengths
        """
        if self.polarization == "s":
            # s-polarization: T = F_out_inv * F_in
            F_in = self._half_transfer_matrix_s(self.eps_in, self.kz_in)
            F_out_inv = self._half_transfer_matrix_s_inv(self.eps_out, self.kz_out)
        elif self.polarization == "p":
            # p-polarization: T = F_out_inv* F_in
            F_in = self._half_transfer_matrix_p(self.eps_in, self.kz_in)
            F_out_inv = self._half_transfer_matrix_p_inv(self.eps_out, self.kz_out)
        else:
            raise ValueError(f"Invalid polarization: {self.polarization}")

        # Vectorized matrix multiplication: T = F_out_out @ F_in
        T = np.matmul(F_out_inv, F_in)

        return T

    def k_vectors(self) -> dict[str, np.ndarray]:
        """
        Get all calculated wave vectors.

        Returns
        -------
        dict
            Dictionary containing k0, kx, kz_in, kz_out arrays
        """
        return {
            "k0": self.k0,
            "kx": self.kx,
            "kz_in": self.kz_in,
            "kz_out": self.kz_out,
        }

    def __repr__(self) -> str:
        """String representation of SingleInterfaceTMatrix object."""
        return (
            f"SingleInterfaceTMatrix(lda={len(self.lda)} wavelengths, "
            f"theta={self.theta * 180 / np.pi:.1f}°, polarization='{self.polarization}')"  # noqa: E501
        )


class LayerPropagationMatrix:
    """
    Layer propagation matrix calculation with wavelength array support.

    Computes the 2x2 propagation (phase accumulation) matrix for a single layer
    at multiple wavelengths with a fixed incident angle and layer thickness.
    The propagation matrix is polarization-independent.

    Parameters
    ----------
    lda : array_like
        Wavelength array in meters, shape (N,)
    theta : float
        Incident angle in degrees (single value, not array)
    d : float
        Layer thickness in meters (scalar)
    eps : float or array_like
        Layer permittivity (scalar or shape (N,))

    Attributes
    ----------
    lda : np.ndarray
        Wavelength array in meters, shape (N,)
    theta : float
        Incident angle in radians
    d : float
        Layer thickness in meters
    eps : np.ndarray
        Layer permittivity, shape (N,)
    k0 : np.ndarray
        Free space wave vectors, shape (N,)
    kx : np.ndarray
        Tangential wave vectors, shape (N,)
    kz : np.ndarray
        z-component wave vectors, shape (N,)

    Examples
    --------
    >>> lda = np.array([500e-9, 600e-9, 700e-9])  # 3 wavelengths
    >>> theta = 30.0  # 30 degrees
    >>> d = 100e-9  # 100 nm layer
    >>> eps = 2.25  # Glass (constant)
    >>> pm = LayerPropagationMatrix(lda, theta, d, eps)
    >>> P_matrices = pm.propagation_matrix()  # Shape (3, 2, 2)
    """

    def __init__(
        self,
        lda: float | np.ndarray,
        theta: float,
        d: float,
        eps: float | np.ndarray,
    ):
        """
        Initialize layer propagation matrix calculator.

        Parameters
        ----------
        lda : array_like
            Wavelength array in meters
        theta : float
            Incident angle in degrees
        d : float
            Layer thickness in meters
        eps : float or array_like
            Layer permittivity
        """
        # Validate and convert inputs
        self._validate_inputs(lda, theta, d, eps)

        # Convert to arrays and ensure proper shapes
        self.lda = np.asarray(lda, dtype=float)
        if self.lda.ndim == 0:
            self.lda = self.lda.reshape(1)

        self.theta = theta * np.pi / 180  # Convert to radians
        self.d = d

        # Broadcast permittivity to match wavelength array shape
        N = len(self.lda)
        self.eps = np.broadcast_to(eps, N)

        # Calculate wave vectors
        self.k0 = 2 * np.pi / self.lda
        self.kx = self.k0 * np.sqrt(self.eps + 1j * 0) * np.sin(self.theta)

        # Calculate z-component of wave vector
        self.kz = self._calculate_kz(self.eps)

    def _validate_inputs(self, lda, theta, d, eps):
        """Validate input parameters."""
        # Validate wavelength
        lda_array = np.asarray(lda)
        if lda_array.ndim > 1:
            raise ValueError("Wavelength array must be 1D")
        if np.any(lda_array <= 0):
            raise ValueError("All wavelengths must be positive")

        # Validate angle
        if np.isclose(theta, 90.0):
            raise ValueError(
                "Incident angle of exactly 90° (grazing incidence) is not supported due to numerical instability"  # noqa: E501
            )
        if theta < 0 or theta > 90:
            warnings.warn(
                "Incident angle should be in range [0, 90) degrees; \
                near 90° results may be unstable",
                stacklevel=2,  # noqa: E501
            )

        # Validate layer thickness
        if d < 0:
            raise ValueError("Layer thickness must be non-negative")

        # Validate permittivity
        eps_array = np.asarray(eps)
        if eps_array.ndim > 1:
            raise ValueError("eps must be scalar or 1D array")
        if eps_array.size > 1 and eps_array.size != lda_array.size:
            raise ValueError("eps length must match wavelength array length")

    def _calculate_kz(self, eps: np.ndarray) -> np.ndarray:
        """
        Calculate z-component of wave vector with proper branch selection.

        Parameters
        ----------
        eps : np.ndarray
            Permittivity array, shape (N,)

        Returns
        -------
        np.ndarray
            z-component of wave vector, shape (N,)
        """
        kz_squared = self.k0**2 * eps - self.kx**2
        # Add small imaginary part for numerical stability
        kz = np.sqrt(kz_squared + 1j * 1e-30)
        return kz

    def propagation_matrix(self) -> np.ndarray:
        """
        Calculate propagation matrices for the layer at all wavelengths.

        The propagation matrix represents phase accumulation through the layer:
        P = [[exp(i*kz*d), 0], [0, exp(-i*kz*d)]]

        Returns
        -------
        np.ndarray
            Propagation matrices, shape (N, 2, 2) where N is number of wavelengths
        """
        N = len(self.lda)
        P = np.zeros((N, 2, 2), dtype=complex)

        # P[0,0] = exp(i*kz*d), P[1,1] = exp(-i*kz*d)
        P[:, 0, 0] = np.exp(1j * self.kz * self.d)
        P[:, 1, 1] = np.exp(-1j * self.kz * self.d)

        return P

    def k_vectors(self) -> dict[str, np.ndarray]:
        """
        Get all calculated wave vectors.

        Returns
        -------
        dict
            Dictionary containing k0, kx, kz arrays
        """
        return {"k0": self.k0, "kx": self.kx, "kz": self.kz}

    def __repr__(self) -> str:
        """String representation of LayerPropagationMatrix object."""
        return (
            f"LayerPropagationMatrix(lda={len(self.lda)} wavelengths, "
            f"theta={self.theta * 180 / np.pi:.1f}°, d={self.d * 1e9:.1f}nm)"
        )
