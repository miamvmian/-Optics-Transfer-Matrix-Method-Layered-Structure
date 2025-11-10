import numpy as np
from TMatrix import TMatrix, RsTsRpTp

from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 10})

# input parameters
lda0 = 1.55e-6
k0 = 2 * np.pi / lda0
eps_metal = -117 + 1j * 8.7
# eps_real = -117
# eps_img = 4.77
# eps_real = -4
# eps_img = 3.6
#
# eps_metal = eps_real + 1j * eps_img
d_metal = 0
# d_metal = 30E-9/lda0

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

# eps_i = [1,4,eps_metal]
# d_i = [0,1/8,d_metal]
#
# eps_i = [1,4,16,eps_metal]
# d_i = [0,1/8,1/16,d_metal]

d_i = [di * lda0 for di in d_i]

# # reverse the incident direction
# eps_i.reverse()
# d_i.reverse()

## scanning in incident angles
thetas = np.arange(0, 90, 0.1)
# thetas=np.array([0])
# lda=1.69E-6
# lda = 1.68819213e-06
# lda = 2.57e-06
# lda2 = 2.0e-6
# lda3 = 1.4e-6
lda = 1.74e-06

t_matrix_s = TMatrix(lda=lda, theta=thetas, pol="s", d_input=d_i, eps_input=eps_i)
rt_s = RsTsRpTp(t_matrix_s)
Rs = rt_s.R()
Ts = rt_s.T()


t_matrix_p = TMatrix(lda=lda, theta=thetas, pol="p", d_input=d_i, eps_input=eps_i)
rt_p = RsTsRpTp(t_matrix_p)
Rp = rt_p.R()
Tp = rt_p.T()
#
# t_matrix_s2 = TMatrix(lda=lda2, theta=thetas, pol='s', d_input=d_i, eps_input=eps_i)
# rt_s2 = RsTsRpTp(t_matrix_s2)
# Rs2 = rt_s2.R()
#
# t_matrix_s3 = TMatrix(lda=lda3, theta=thetas, pol='s', d_input=d_i, eps_input=eps_i)
# rt_s3 = RsTsRpTp(t_matrix_s3)
# Rs3 = rt_s3.R()

# fig, ax = plt.subplots(1,1,figsize=(20, 15))
fig, ax = plt.subplots(1, 1)
ax.plot(thetas, Rs, color="r", label=f"$R_s$", linewidth=1)
ax.plot(thetas, Ts, color="r", label=r"$T_s$", linestyle="dashed", linewidth=1)

# ax.plot(thetas, Rs2, color='g', label=f"$lda={lda2*1.0e6:0.2f}$", linewidth=1)
# # ax.plot(thetas, 1-Rs, color='r', label=r"$A_s$", linestyle="dashed", linewidth=1)
#
# ax.plot(thetas, Rs3, color='k', label=f"$lda={lda3*1.0e6:0.2f}$", linewidth=1)

ax.plot(thetas, Rp, color="g", label=r"$R_p$", linewidth=1)
ax.plot(thetas, Tp, color="g", label=r"$T_p$", linestyle="dashed", linewidth=1)
#
# ax.plot(thetas, Rp, color='g', label=r"$R_p$", linewidth=1)
# ax.plot(thetas, 1-Rp, color='g', label=r"$A_p$", linestyle="dashed", linewidth=1)

ax.set_xlabel(f"Incident Angle($^o$)")
# ax.set_ylabel("Reflectance/Absorptance")
ax.set_ylabel("Reflectance $R_s$")
# ax.set_title(f"Reflectance and Transmittance:$\lambda$={ratio}$\lambda_0$")
ax.set_title(f"Reflectance and Absorptance @wavelength {lda * 1e6:.2f}um")

# Adding legend, which helps us recognize the curve according to it's color
ax.legend()
# To load the display window
# plt.ylim([0,1.2])
plt.savefig("Reflectance and Absorptance S- and P-mode.png")
# plt.show()


## Total=Transmittance+Reflectance
tot_s = np.array(Rs) + np.array(Ts)
tot_p = np.array(Rp) + np.array(Tp)
# fig, ax = plt.subplots(1,1,figsize=(20, 15))
fig2, ax2 = plt.subplots(1, 1)
ax2.plot(thetas, tot_s, color="r", label=r"s-mode", linewidth=2)
ax2.plot(thetas, tot_p, color="g", label=r"p-mode", linewidth=1)


ax2.set_xlabel("Incident Angle($^o$)DEG")
ax2.set_ylabel("Reflectance+Transmittance")
# ax.set_title(f"Reflectance and Transmittance:$\lambda$={ratio}$\lambda_0$")
# ax2.set_title(f"Total(Reflectance+Transmittance)@{lda*1E6:.2f}um")
ax2.set_title(
    f"Total(Reflectance+Transmittance)@{lda * 1e6:.2f}um, $\\epsilon$={eps_metal}"
)


# Adding legend, which helps us recognize the curve according to it's color
ax2.legend()
# To load the display window
plt.ylim([0.999, 1.001])
plt.savefig("Total.png")
# plt.show()


if __name__ == "__main__":
    pass
