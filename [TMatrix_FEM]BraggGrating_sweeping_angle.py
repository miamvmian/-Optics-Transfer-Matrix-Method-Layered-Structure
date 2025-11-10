import numpy as np
from TMatrix import TMatrix, RsTsRpTp

from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 10})

# input parameters
lda0 = 1.55e-6
k0 = 2 * np.pi / lda0

eps_metal = 1

d_metal = 0

# 10 dielectric Interfaces + Au
eps_i = [1, 4, 16, 4, 16, 4, 16, 4, 16, 4, 16, eps_metal]
d_i = [
    0,
    1 / 8,
    1 / 16,
    1 / 8,
    1 / 16,
    1 / 8,
    1 / 16,
    1 / 8,
    1 / 16,
    1 / 8,
    1 / 16,
    d_metal,
]

d_i = [di * lda0 for di in d_i]

# # # reverse the incident direction
# eps_i.reverse()
# d_i.reverse()

# Sweeping in wavelength

thetas = 0

# lda_l = np.linspace(lda0*0.6,lda0*1.5,1500)
lda_l = np.arange(1.0, 1.6, 0.002) * 1e-6


def R_T_wl(theta, lda_l, d_i, eps_i):
    RRs = []
    TTs = []
    RRp = []
    TTp = []

    for ldai in lda_l:
        t_matrix_s = TMatrix(
            lda=ldai, theta=theta, pol="s", d_input=d_i, eps_input=eps_i
        )
        rt_s = RsTsRpTp(t_matrix_s)
        Rs = rt_s.R()
        Ts = rt_s.T()
        RRs.append(Rs)
        TTs.append(Ts.real)

        t_matrix_p = TMatrix(
            lda=ldai, theta=theta, pol="p", d_input=d_i, eps_input=eps_i
        )
        rt_p = RsTsRpTp(t_matrix_p)
        Rp = rt_p.R()
        Tp = rt_p.T()
        RRp.append(Rp)
        TTp.append(Tp.real)
    return RRs, RRp, TTs, TTp


RRs, RRp, TTs, TTp = R_T_wl(thetas, lda_l, d_i, eps_i)


## Data process from FEM
import pandas as pd

df = pd.read_csv(
    "COMSOL_data/wavelength_sweeping/Reflectance_TM.csv",
    on_bad_lines="skip",
    header=None,
)
df_R = df[7:]
df = pd.read_csv(
    "COMSOL_data/wavelength_sweeping/Transmittance_TM.csv",
    on_bad_lines="skip",
    header=None,
)
df_T = df[7:]


df_R = df_R.astype("float64")
df_T = df_T.astype("float64")

df_wl = df_R[0].values

# fig, ax = plt.subplots(1,1,figsize=(20, 15))


###### TE mode
# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(lda_l*1E6, RRs, color='k', label=r"$R_{TE}$ (Tmatrix)", linewidth=1)
# ax1.plot(df_wl, df_R[1].values, color='r',linestyle="dashed",label=r"$R_{TE}$ (FEM)", linewidth=2)
# ax1.set_ylabel("Reflectance")
# ax1.set_title(f"TE mode Transfer Matrix VS FEM")
# ax1.legend()

# ax2.plot(lda_l*1E6, TTs, color='k', label=r"$T_{TE}$ (Tmatrix)", linewidth=1)
# ax2.plot(df_wl, df_T[1].values, color='r',linestyle="dashed",label=r"$T_{TE}$ (FEM)", linewidth=2)
#
# ax2.set_xlabel(r"Wavelength($\mu$m)")
# ax2.set_ylabel("Transmittance")
# ax2.legend()
# plt.savefig("TMatrix_VS_FEM_TE.png")


####### TM mode
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(lda_l * 1e6, RRs, color="k", label=r"$R_{TM}$ (Tmatrix)", linewidth=1)
ax1.plot(
    df_wl,
    df_R[1].values,
    color="r",
    linestyle="dashed",
    label=r"$R_{TM}$ (FEM)",
    linewidth=2,
)
ax1.set_ylabel("Reflectance")
ax1.set_title(f"TM mode Transfer Matrix VS FEM")
ax1.legend()

ax2.plot(lda_l * 1e6, TTs, color="k", label=r"$T_{TM}$ (Tmatrix)", linewidth=1)
ax2.plot(
    df_wl,
    df_T[1].values,
    color="r",
    linestyle="dashed",
    label=r"$T_{TM}$ (FEM)",
    linewidth=2,
)

ax2.set_xlabel(r"Wavelength($\mu$m)")
ax2.set_ylabel("Transmittance")
ax2.legend()

plt.savefig("TMatrix_VS_FEM_TM.png")


## Total=Transmittance+Reflectance
# tot_s = np.array(RRs) + np.array(TTs)
# tot_p = np.array(RRp) + np.array(TTp)


if __name__ == "__main__":
    pass
