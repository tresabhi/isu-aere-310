# AERE 310 HW 6
# Abhigyaan Deep

import math
import numpy as np
import os
import matplotlib.pyplot as plt
import concurrent.futures
from scipy.interpolate import interp1d
from tabulate import tabulate

N = 150  # sections
epsilon = 1e-8  # threshold
beta = 10**-2.5  # damping

V_infinity = 65

S = 17
AR = 8
lambda_ = 0.7
b = math.sqrt(S * AR)
c_r = 2 * S / (b * (1 + lambda_))
c_t = lambda_ * c_r

lift_plot_alpha_0_deg = -17
lift_plot_alpha_1_deg = 17
lift_plot_d_alpha_deg = 1
lift_plot_alphas_deg = np.linspace(
    lift_plot_alpha_0_deg,
    lift_plot_alpha_1_deg,
    int((lift_plot_alpha_1_deg - lift_plot_alpha_0_deg) / lift_plot_d_alpha_deg) + 1,
)

slope_alpha_0_deg = -8
slope_alpha_1_deg = 8
slope_alpha_0 = slope_alpha_0_deg * math.pi / 180
slope_alpha_1 = slope_alpha_1_deg * math.pi / 180

gamma_plot_alpha_0_deg = -20
gamma_plot_alpha_1_deg = 20
gamma_plot_d_alpha_deg = 5
gamma_plot_alphas_deg = np.linspace(
    gamma_plot_alpha_0_deg,
    gamma_plot_alpha_1_deg,
    int((gamma_plot_alpha_1_deg - gamma_plot_alpha_0_deg) / gamma_plot_d_alpha_deg) + 1,
)

# incidence angles at root and tip
i_r = 1.5 * math.pi / 180
i_t = -1.5 * math.pi / 180

delta_z = b / N


"""
got sick of jumping between workspaces so now just place airfoil.dat adjacent
to this script :)
"""
airfoil_deg = np.loadtxt(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "airfoil.dat")
)
airfoil = airfoil_deg.copy()
airfoil[:, 0] = np.deg2rad(airfoil[:, 0])

c_l_interp = interp1d(
    airfoil[:, 0], airfoil[:, 1], fill_value="extrapolate", kind="cubic"
)


def c_l(alpha):
    return c_l_interp(alpha)


def c(z):
    x = abs(z) / (b / 2)
    return c_r * (1 - x) + c_t * x


def D(x):
    return 4 / (1 - 4 * x**2)


def alpha_i_i(Gammas, i):
    return (
        -1
        / (4 * math.pi * V_infinity * delta_z)
        * sum(Gammas[j - 1] * D(i - j) for j in range(1, N + 1))
    )


def iterate(Gammas, alpha):
    Gammas_new = []

    for i in range(1, N + 1):
        z_i = -b / 2 + (i - 0.5) * delta_z
        alpha_eff_i = alpha + incidence(z_i) + alpha_i_i(Gammas, i)

        c_i = c(z_i)
        C_l_i = c_l(alpha_eff_i)

        Gamma_new_i = (V_infinity * c_i / 2) * C_l_i
        Gamma_old_i = Gammas[i - 1]
        Gamma = Gamma_old_i + beta * (Gamma_new_i - Gamma_old_i)

        Gammas_new.append(Gamma)

    return Gammas_new


# incidence angle interpolator
def incidence(z):
    x = abs(z) / (b / 2)
    return i_r * (1 - x) + i_t * x


c_l_z_0 = c_l(incidence(0))  # c_l of the incidence angle at z = 0
Gamma_0 = V_infinity * c_r / 2 * c_l_z_0


def z(i):
    return -b / 2 + delta_z / 2 + (i - 1) * delta_z


# i just love the word ephemeral; here, it means initial
def Gamma_ephemeral(i):
    return Gamma_0 * math.sqrt(1 - (z(i) / (b / 2)) ** 2)


def converge_Gammas(alpha_deg):
    print(f"🟡 Received request for alpha = {alpha_deg}°")

    alpha = alpha_deg * math.pi / 180
    Gammas = [Gamma_ephemeral(i) for i in range(1, N + 1)]
    converged = False
    iteration = 0

    while not converged:
        Gammas_new = iterate(Gammas, alpha)
        epsilon_i = (
            1
            / N
            * sum(
                ((Gammas_new[i - 1] - Gammas[i - 1]) / Gammas[i - 1]) ** 2
                for i in range(1, N + 1)
            )
        )

        converged = epsilon_i < epsilon
        Gammas = Gammas_new
        iteration += 1

    print(
        f"🟢 Converged alpha = {round(alpha * 180 / math.pi)}° in {iteration} iterations"
    )

    return Gammas


def render_coefficients(C_Ls, C_D_is):
    _, ax1 = plt.subplots()

    ax1.plot(
        airfoil_deg[:, 0],
        airfoil_deg[:, 1],
        color="tab:orange",
        label="C_l",
    )
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(lift_plot_alphas_deg, C_D_is, color="tab:red", label="C_D_i")
    ax2.set_ylabel("C_D_i", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="lower right")

    ax1.plot(lift_plot_alphas_deg, C_Ls, color="tab:blue", label="C_L")
    ax1.set_xlabel("Alpha (degrees)")
    ax1.set_ylabel("C_L", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    plt.title("C_L and C_D_i vs Alpha")
    plt.show()


def render_Gammas():
    zs = [z(i) for i in range(1, N + 1)]

    _, ax = plt.subplots()

    for alpha_deg in gamma_plot_alphas_deg:
        Gammas = Gamma_cache[alpha_deg] / Gamma_0
        ax.plot(zs, Gammas, label=f"{alpha_deg}°")

    plt.title("Gamma vs Z")
    plt.show()


# i made a cache because i need gammas again for the second plot
Gamma_cache = {}


def cache_Gammas():
    alphas_deg = list(set([*lift_plot_alphas_deg, *gamma_plot_alphas_deg]))

    print(
        f"🔵 Spawning {os.cpu_count()} asynchronous compute threads for {len(alphas_deg)} Gammas..."
    )
    with concurrent.futures.ProcessPoolExecutor() as executor:
        Gammas_array = executor.map(converge_Gammas, alphas_deg)

    for alpha_deg, Gammas in zip(alphas_deg, Gammas_array):
        Gamma_cache[alpha_deg] = Gammas


def main():
    cache_Gammas()
    render_Gammas()

    C_Ls = []
    C_D_is = []

    for alpha_deg in lift_plot_alphas_deg:
        Gammas = Gamma_cache[alpha_deg]
        C_L = (
            2 * delta_z / (V_infinity * S) * sum(Gammas[i - 1] for i in range(1, N + 1))
        )
        C_D_i = (
            -2
            * delta_z
            / (V_infinity * S)
            * sum(
                Gammas[i - 1] * math.sin(alpha_i_i(Gammas, i)) for i in range(1, N + 1)
            )
        )
        C_Ls.append(C_L)
        C_D_is.append(C_D_i)

    c_l_0 = c_l(slope_alpha_0)
    c_l_1 = c_l(slope_alpha_1)
    C_L_0 = C_Ls[np.where(slope_alpha_0_deg == lift_plot_alphas_deg)[0][0]]
    C_L_1 = C_Ls[np.where(slope_alpha_1_deg == lift_plot_alphas_deg)[0][0]]

    a_0 = (c_l_1 - c_l_0) / (slope_alpha_1 - slope_alpha_0)
    a = (C_L_1 - C_L_0) / (slope_alpha_1 - slope_alpha_0)

    print(
        tabulate(
            [["a_0 (airfoil slope)", a_0], ["a (3D wing slope)", a]],
        )
    )

    render_coefficients(C_Ls, C_D_is)


if __name__ == "__main__":
    main()
