# defining class TMatrix
# The transmittance of planar layered structure is
#
from collections.abc import Iterable
from typing import List

import numpy as np
from numpy.linalg import multi_dot  # matrix multiplication


class TMatrix_Basic:
    def __init__(self, lda, theta):
        self.lda = lda
        self.k0 = 2 * np.pi / lda
        self.theta = theta * np.pi / 180  # incident angle
        self.kx = self.k0 * np.sin(self.theta)

    def _Fs(self, eps):
        kz = np.sqrt(self.k0**2 * eps - self.kx**2 + 1j * 0)
        return np.array([[1, 1], [-kz / self.k0, kz / self.k0]])

    def _Fs_Invs(self, eps):
        kz = np.sqrt(self.k0**2 * eps - self.kx**2 + 1j * 0)
        return 1 / 2 * np.array([[1, -self.k0 / kz], [1, self.k0 / kz]])

    def _Fp(self, eps):
        kz = np.sqrt(self.k0**2 * eps - self.kx**2 + 1j * 0)
        return np.array([[kz / (self.k0 * eps), -kz / (self.k0 * eps)], [1, 1]])

    def _Fp_Invs(self, eps):
        kz = np.sqrt(self.k0**2 * eps - self.kx**2 + 1j * 0)
        return 1 / 2 * np.array([[(self.k0 * eps) / kz, 1], [-(self.k0 * eps) / kz, 1]])

    def _P_phase(self, t, eps):  # phase accumulation due to propagatation
        kz = np.sqrt(self.k0**2 * eps - self.kx**2 + 1j * 0)
        return np.array([[np.exp(1j * kz * t), 0], [0, np.exp(-1j * kz * t)]])


class TMatrix(TMatrix_Basic):
    def __init__(self, lda, theta, pol, d_input, eps_input):
        super().__init__(lda, theta)
        self.theta = theta * np.pi / 180  # incident angle
        self.pol = pol

        if len(d_input) != len(eps_input):
            print("Input thinkness and permittivity do not match in dimension")
            return

        self.thickness = d_input
        self.eps = eps_input

        if isinstance(theta, Iterable):
            self.Tmat = []
            self.kx_angles = (
                self.k0 * np.sin(self.theta) * np.sqrt(self.eps[0] + 1j * 0)
            )
            for kx_ in self.kx_angles:
                self.kx = kx_
                self.Tmat.append(self.run())

            return

        else:
            self.kx = self.k0 * np.sin(self.theta) * np.sqrt(self.eps[0] + 1j * 0)

        self.Tmat = self.run()

    def __repr__(self):
        return f"TMatrix(lda={self.lda * 1e6:0.2f}um,pol={self.pol})"

    def start_layer(self, t_0, eps_0):
        t = t_0
        eps = eps_0

        if self.pol == 1 or self.pol in "sS":
            Ts = self._P_phase(t, eps)
            Fs = self._Fs(eps)
            return np.dot(Fs, Ts)

        elif self.pol == 2 or self.pol in "pP":
            Tp = self._P_phase(t, eps)
            Fp = self._Fp(eps)
            return np.dot(Fp, Tp)

        else:
            print("Input wrong polirization indicator.")
            return

    def end_layer(self, t_end, eps_end):
        t = t_end
        eps = eps_end

        if self.pol == 1 or self.pol in "sS":
            Ts = self._P_phase(t, eps)
            Fs_invs = self._Fs_Invs(eps)
            return np.dot(Ts, Fs_invs)

        elif self.pol == 2 or self.pol in "pP":
            Tp = self._P_phase(t, eps)
            Fp_invs = self._Fp_Invs(eps)
            return np.dot(Tp, Fp_invs)

    def single_layer(self, t, eps):
        if self.pol == 1 or self.pol in "sS":
            return multi_dot([self._Fs(eps), self._P_phase(t, eps), self._Fs_Invs(eps)])
        elif self.pol == 2 or self.pol in "pP":
            return multi_dot([self._Fp(eps), self._P_phase(t, eps), self._Fp_Invs(eps)])

    def middle_layers(self):
        if len(self.thickness) == 2:
            return np.eye(2, 2)

        ml_t = self.thickness[1:-1]
        ml_eps = self.eps[1:-1]

        layers = [self.single_layer(t, eps) for t, eps in zip(ml_t, ml_eps)]
        if len(layers) == 1:
            return layers[0]
        layers.reverse()
        return multi_dot(layers)

    def full_layers(self):
        t_s = self.thickness[0]
        t_e = self.thickness[-1]
        eps_s = self.eps[0]
        eps_e = self.eps[-1]

        return multi_dot(
            [
                self.end_layer(t_e, eps_e),
                self.middle_layers(),
                self.start_layer(t_s, eps_s),
            ]
        )

    def run(self):
        return self.full_layers()


class RsTsRpTp:
    def __init__(self, obj):
        if isinstance(obj, TMatrix):
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

            # self.cosTheta_1 = self.kz[0]/self.k0/np.sqrt(self.eps_s)
            # self.cosTheta_N = self.kz[-1]/self.k0/np.sqrt(self.eps_e)

        else:
            print("input an instance of class TMatrix")

    def __ts(self):
        if not isinstance(self.Tmat, List):
            T_ = self.Tmat
            return T_[0, 0] - T_[1, 0] / T_[1, 1] * T_[0, 1]

        else:
            return [T_[0, 0] - T_[1, 0] / T_[1, 1] * T_[0, 1] for T_ in self.Tmat]

    def __rs(self):
        if isinstance(self.Tmat, List):
            return [-T_[1, 0] / T_[1, 1] for T_ in self.Tmat]

        else:
            T_ = self.Tmat
            return -T_[1, 0] / T_[1, 1]

    def __rp(self):
        if isinstance(self.Tmat, List):
            return [T_[1, 0] / T_[1, 1] for T_ in self.Tmat]
        else:
            T_ = self.Tmat
            return T_[1, 0] / T_[1, 1]

    def __tp(self):
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

    def r(self):
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
            print("please indicate s/p mode")

    def t(self):
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
            print("please indicate s/p mode")

    def R(self):
        r = self.r()
        if isinstance(r, Iterable):
            return np.array([np.abs(r_) ** 2 for r_ in r])
        else:
            return np.abs(r) ** 2

    def T(self):
        t = self.t()
        if isinstance(t, Iterable):
            # return np.array([self.cosTheta_N[i]/self.cosTheta_1[i]
            #                     *np.abs(t_*np.sqrt(self.eps_e)/np.sqrt(self.eps_s))**2 for i, t_ in enumerate(t)])
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
            # return self.cosTheta_N_/self.cosTheta_1*np.abs(t)**2
            if self.pol == 1 or self.pol in "sS":
                return self.n_cosTheta_N_TE / self.cosTheta_1 * np.abs(t) ** 2
            elif self.pol == 2 or self.pol in "pP":
                return self.n_cosTheta_N_TM / self.cosTheta_1 * np.abs(t) ** 2
