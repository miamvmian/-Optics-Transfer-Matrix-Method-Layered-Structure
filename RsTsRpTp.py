# The transmittance of planar layered structure is calculated using
# the Transfer Matrix Method for electromagnetic wave propagation
import numpy as np
from typing import List, Union, Iterable
from TMatrix import TMatrix


class RsTsRpTp:
    """
    Calculate reflection and transmission coefficients from transfer matrix.

    This class extracts the physical reflection (r) and transmission (t)
    coefficients, as well as reflectance (R) and transmittance (T) from
    the transfer matrix.

    Attributes
    ----------
    obj : TMatrix
        TMatrix object containing the transfer matrix
    Tmat : np.ndarray or list
        Transfer matrix or list of transfer matrices
    pol : str or int
        Polarization mode
    eps : list
        Layer permittivities
    theta : float or array
        Incident angle(s)

    Examples
    --------
    >>> tm = TMatrix(1.55e-6, 0, 's', [0, 100e-9, 0], [1, 2.25, 1])
    >>> rt = RsTsRpTp(tm)
    >>> R = rt.R()  # Reflectance
    >>> T = rt.T()  # Transmittance
    >>> print(f"R + T = {R + T}")  # Should be ~1 for lossless
    """

    def __init__(self, obj: TMatrix):
        """
        Initialize reflection/transmission calculator.

        Parameters
        ----------
        obj : TMatrix
            TMatrix object to analyze

        Raises
        ------
        TypeError
            If obj is not a TMatrix instance
        """
        if not isinstance(obj, TMatrix):
            raise TypeError("Input must be an instance of class TMatrix")

        self.obj = obj
        self.Tmat = self.obj.Tmat
        self.pol = self.obj.pol
        self.eps = self.obj.eps
        self.theta = self.obj.theta

        self.eps_start_sqrt = np.sqrt(self.eps[0])
        self.eps_end_sqrt = np.sqrt(self.eps[-1])

        self.k0 = self.obj.k0
        self.kx_sq = self.k0**2 * self.eps[0] * np.sin(self.theta) ** 2
        self.kz = [
            np.sqrt(self.k0**2 * epsi - self.kx_sq + 1j * 0) for epsi in self.eps
        ]

        self.cosTheta_1 = self.kz[0] / self.k0 / self.eps_start_sqrt
        self.cosTheta_N = self.kz[-1] / self.k0 / self.eps_end_sqrt

        # self.n_cosTheta_N_TE represents refractive index (n) multiply
        # n*cos(theta_N), where theta_N is the refraction angle
        self.n_cosTheta_N_TE = self.cosTheta_N * self.eps_end_sqrt
        self.n_cosTheta_N_TE = self.n_cosTheta_N_TE.real

        self.n_cosTheta_N_TM = self.cosTheta_N * np.conj(self.eps_end_sqrt)
        self.n_cosTheta_N_TM = self.n_cosTheta_N_TM.real

    def __ts(self) -> Union[complex, np.ndarray]:
        """
        Calculate s-polarization transmission coefficient.

        Returns
        -------
        complex or np.ndarray
            Transmission coefficient(s) for s-polarization
        """
        if not isinstance(self.Tmat, List):
            T_ = self.Tmat
            return T_[0, 0] - T_[1, 0] / T_[1, 1] * T_[0, 1]
        else:
            return [T_[0, 0] - T_[1, 0] / T_[1, 1] * T_[0, 1] for T_ in self.Tmat]

    def __rs(self) -> Union[complex, np.ndarray]:
        """
        Calculate s-polarization reflection coefficient.

        Returns
        -------
        complex or np.ndarray
            Reflection coefficient(s) for s-polarization
        """
        if isinstance(self.Tmat, List):
            return [-T_[1, 0] / T_[1, 1] for T_ in self.Tmat]
        else:
            T_ = self.Tmat
            return -T_[1, 0] / T_[1, 1]

    def __rp(self) -> Union[complex, np.ndarray]:
        """
        Calculate p-polarization reflection coefficient.

        Returns
        -------
        complex or np.ndarray
            Reflection coefficient(s) for p-polarization
        """
        if isinstance(self.Tmat, List):
            return [T_[1, 0] / T_[1, 1] for T_ in self.Tmat]
        else:
            T_ = self.Tmat
            return T_[1, 0] / T_[1, 1]

    def __tp(self) -> Union[complex, np.ndarray]:
        """
        Calculate p-polarization transmission coefficient.

        Returns
        -------
        complex or np.ndarray
            Transmission coefficient(s) for p-polarization
        """
        if isinstance(self.Tmat, List):
            return [
                np.sqrt(self.eps[0] / self.eps[-1] + 1j * 0)
                * (T_[0, 0] - T_[1, 0] / T_[1, 1] * T_[0, 1])
                for T_ in self.Tmat
            ]
        else:
            T_ = self.Tmat
            return np.sqrt(self.eps[0] / self.eps[-1] + 1j * 0) * (
                T_[0, 0] - T_[1, 0] / T_[1, 1] * T_[0, 1]
            )

    def r(self) -> Union[complex, np.ndarray]:
        """
        Get reflection coefficient for the current polarization.

        Returns
        -------
        complex or np.ndarray
            Complex reflection coefficient(s)

        Raises
        ------
        ValueError
            If polarization is not properly specified
        """
        if self.pol == 1 or self.pol in "sS":
            rs_ = self.__rs()
            if isinstance(rs_, List):
                return np.array(rs_)
            else:
                return rs_

        elif self.pol == 2 or self.pol in "pP":
            rp_ = self.__rp()
            if isinstance(rp_, List):
                return np.array(rp_)
            else:
                return rp_
        else:
            raise ValueError("Polarization must be specified as s/p mode")

    def t(self) -> Union[complex, np.ndarray]:
        """
        Get transmission coefficient for the current polarization.

        Returns
        -------
        complex or np.ndarray
            Complex transmission coefficient(s)

        Raises
        ------
        ValueError
            If polarization is not properly specified
        """
        if self.pol == 1 or self.pol in "sS":
            ts_ = self.__ts()
            if isinstance(ts_, Iterable):
                return np.array(ts_)
            else:
                return ts_
        elif self.pol == 2 or self.pol in "pP":
            tp_ = self.__tp()
            if isinstance(tp_, Iterable):
                return np.array(tp_)
            else:
                return tp_
        else:
            raise ValueError("Polarization must be specified as s/p mode")

    def R(self) -> Union[float, np.ndarray]:
        """
        Calculate reflectance (power reflection coefficient).

        Returns
        -------
        float or np.ndarray
            Reflectance value(s) in range [0, 1]
        """
        r = self.r()
        if isinstance(r, Iterable):
            return np.array([np.abs(r_) ** 2 for r_ in r])
        else:
            return np.abs(r) ** 2

    def T(self) -> Union[float, np.ndarray]:
        """
        Calculate transmittance (power transmission coefficient).

        This method accounts for the change in wave impedance and
        angle between incident and transmitted media.

        Returns
        -------
        float or np.ndarray
            Transmittance value(s) in range [0, 1]
        """
        t = self.t()
        if isinstance(t, Iterable):
            if self.pol == 1 or self.pol in "sS":
                return np.array(
                    [
                        self.n_cosTheta_N_TE[i] / self.cosTheta_1[i] * np.abs(t_) ** 2
                        for i, t_ in enumerate(t)
                    ]
                ).real
            elif self.pol == 2 or self.pol in "pP":
                return np.array(
                    [
                        self.n_cosTheta_N_TM[i] / self.cosTheta_1[i] * np.abs(t_) ** 2
                        for i, t_ in enumerate(t)
                    ]
                ).real
        else:
            if self.pol == 1 or self.pol in "sS":
                return (self.n_cosTheta_N_TE / self.cosTheta_1 * np.abs(t) ** 2).real
            elif self.pol == 2 or self.pol in "pP":
                return (self.n_cosTheta_N_TM / self.cosTheta_1 * np.abs(t) ** 2).real

    def energy_conservation_check(
        self, tolerance: float = 1e-6
    ) -> Union[bool, np.ndarray]:
        """
        Check if energy is conserved (R + T ≈ 1 for lossless materials).

        Parameters
        ----------
        tolerance : float, optional
            Maximum allowed deviation from 1.0 (default: 1e-6)

        Returns
        -------
        bool or np.ndarray
            True if energy is conserved within tolerance
        """
        R = self.R()
        T = self.T()
        total = R + T

        if isinstance(total, np.ndarray):
            return np.abs(total - 1.0) < tolerance
        else:
            return abs(total - 1.0) < tolerance
