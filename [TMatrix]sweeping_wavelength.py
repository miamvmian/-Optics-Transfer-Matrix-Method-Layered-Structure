import numpy as np
from TMatrix import TMatrix
from RsTsRpTp import RsTsRpTp

from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 10})

# input parameters
lda0 = 1.55e-6
k0 = 2 * np.pi / lda0

# eps_real = -117
# eps_img = 4.77
#
# eps_real = -117
# eps_img = 3.6
#
# eps_metal = eps_real + 1j * eps_img  # the minimum lda = 1.68819213e-06
# eps_metal = -117
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
lda_l = np.linspace(1.0, 1.6, 1500) * 1e-6


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


# Define a function to find the wavelength
# correspondent to the minimum Reflectance
def min_wl_R(refl, wl):
    if not isinstance(refl, np.ndarray):
        refl = np.array(refl)
    if not isinstance(wl, np.ndarray):
        wl = np.array(wl)
    return wl[refl == refl.min()][0], refl.min()


RRs, RRp, TTs, TTp = R_T_wl(thetas, lda_l, d_i, eps_i)

# lda_dip_s = spp_wl(RRs,lda_l)
# lda_dip_p = spp_wl(RRp,lda_l)

#
# eps_img_l = np.linspace(1,100,500)
# lda_dip_p = []
# for eps_imgi  in eps_img_l:
#     eps_img = eps_imgi
#     eps_metal = eps_real + 1j * eps_img
#
#     eps_i = [1, 4, 16, 4, 16, 4, 16, 4, 16, 4, 16, eps_metal]
#     _, RRp, _, _ = R_T_wl(thetas, lda_l, d_i, eps_i)
#     lda_min, RRp_min= min_wl_R(RRp,lda_l)
#     lda_dip_p.append([lda_min,RRp_min])

#
# eps_real_l = -1 * np.linspace(1, 140, 500)
#
# lda_dip_p = []
# for eps_reali in eps_real_l:
#     eps_real = eps_reali
#     eps_metal = eps_real + 1j * eps_img
#
#     eps_i = [1, 4, 16, 4, 16, 4, 16, 4, 16, 4, 16, eps_metal]
#     _, RRp, _, _ = R_T_wl(thetas, lda_l, d_i, eps_i)
#     lda_min, RRp_min = min_wl_R(RRp, lda_l)
#     lda_dip_p.append([lda_min, RRp_min])

# print(lda_dip_p)
#
fig, ax = plt.subplots(1, 1, figsize=(20, 15))
# fig, (ax1, ax2) = plt.subplots(2, 1)
ax.plot(lda_l * 1e6, RRs, color="g", label=r"$R_s$", linewidth=2)
# ax.plot(lda_l, TTs, color='g', label=r"$Re(T_s)$", linestyle="dashed", linewidth=2)
# ax.plot(lda_l*1E6, RRp, color='r', label=r"$R_p$", linewidth=1)
# ax.plot(lda_l, TTp, color='r', label=r"$Re(T_p)$", linestyle="dashed", linewidth=1)
ax.set_xlabel(r"Wavelength($\mu$m)")
ax.set_ylabel("Reflectance")
ax.set_title(f"Reflectance of TE mode")

# plt.xlim([1e-6, 1.6e-6])
plt.ylim([0.0, 1])

plt.savefig("Reflectance.png")

#
# ax.set_xlabel(r"Wavelength($\mu$m)")
# ax.set_ylabel("Reflectance")
# ax.plot(thetas, Rp, color='g', label=r"$R_p$", linewidth=2)
# ax.plot(thetas, Tp.real, color='g', label=r"$Re(T_p)$", linestyle="dashed", linewidth=2)
#
# ax.plot(lda_l*1E6, RRp, color='r', label=r"$R_p$", linewidth=1)
# ax.plot(lda_l*1E6, TTp, color='g', label=r"$Re(T_p)$", linestyle="dashed", linewidth=1)
# ax.set_xlabel(r"Wavelength($\mu$m)")
# ax.set_ylabel("Reflectance/Transmittance")
# ax.set_title(f"Reflectance: $\\epsilon_m$=-117+j{eps_img} @incdience {thetas}DEG")

# ax1.scatter(eps_img_l, np.array(lda_dip_p).transpose()[1], color='r', label=r"$R_p$", linewidth=1,s=0.5)
# # ax1.set_xlabel(r"$\epsilon_{real}$")
# ax1.set_ylabel("Minimum Reflectance")
# ax1.set_title(f"Reflectance of p-mode: $\\epsilon_m$={eps_real}+j$\\epsilon_i$$_m$$_g$ @incdience {thetas}DEG")
# # ax1.set_title(f"Reflectance of p-mode: $\\epsilon_m$=$\\epsilon_r$$_e$$_a$$_l$+j{eps_img} @incdience {thetas}DEG")
# ax1.tick_params(axis='x', which='both')
# plt.ylim([-0.01, 0.1])
#
# ax2.scatter(eps_img_l, np.array(lda_dip_p).transpose()[0] * 1E6, color='r', label=r"$R_p$", linewidth=1,s=0.5)
# ax2.set_xlabel(r"$\epsilon_{real}$")
# ax2.set_ylabel(r"Wavelength $\mu$m")
# # ax2.set_title(f"Reflectance minimum with $\\epsilon_m$=$\\epsilon_r$$_e$$_a$$_l$+j{eps_img} @incdience {thetas}DEG")
# # ax.set_title(f"Reflectance dip: S@{lda_dip_s[0]*1E6:.2f}um; P@{lda_dip_p[0]*1E6:.2f}um at {thetas}DEG")
#
# # Adding legend, which helps us recognize the curve according to it's color
# # ax2.legend()
# # To load the display window
# plt.ylim([1.6, 3.0])
# plt.savefig("Reflectance.png")
# # plt.show()


## Total=Transmittance+Reflectance
tot_s = np.array(RRs) + np.array(TTs)
tot_p = np.array(RRp) + np.array(TTp)

tot_p_dip = min_wl_R(tot_p, lda_l)
# spp_dip_p = R_T_wl(tot_p,lda_l)

# fig, ax = plt.subplots(1,1,figsize=(20, 15))
fig3, ax3 = plt.subplots(1, 1)
ax3.plot(
    lda_l * 1e6, tot_s, color="r", label=r"$s-mode$", linewidth=2, linestyle="dashed"
)
ax3.plot(lda_l * 1e6, tot_p, color="g", label=r"p-mode", linewidth=1)
ax3.set_title(f"Reflectance+Transmittance")
#
# ax2.plot(lda_l*1E6, RRs, color='r', label=r"s-mode", linewidth=2)
# ax2.plot(lda_l*1E6, RRp, color='g', label=r"p-mode", linewidth=1)
#
#
ax3.set_xlabel(r"Wavelength($\mu$m)")
ax3.set_ylabel("Reflectance/Transmittance")
# ax.set_title(f"Reflectance and Transmittance:$\lambda$={ratio}$\lambda_0$")
# ax2.set_title(f"$R_p$+$Re(T_p)$ @{thetas}DEG Minimum@{spp_dip_p[0]*1E6:.2f}um")
# ax2.set_title(f"$R_p$ and $R_s$ @{thetas}DEG, $\\epsilon_m$={eps_metal}")


# ax2.scatter(eps_img_l, np.array(lda_dip_p).transpose()[0]*1E6, color='r', label=r"$R_p$", s=0.5)
# ax2.set_xlabel(r"$\epsilon_{img}$")
# ax2.set_ylabel(r"Wavelength($\mu$m)")
# ax2.set_title(f"$\\epsilon_m$=-117+j$\\epsilon_i$$_m$$_g$ @incdience {thetas}DEG")
#

# Adding legend, which helps us recognize the curve according to it's color
ax3.legend()
# To load the display window
# plt.ylim([0,1.1])
# plt.ylim([0.988, 1.01])
plt.savefig("Total.png")
# plt.show()


if __name__ == "__main__":
    pass
