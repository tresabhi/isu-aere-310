# AERE 310 HW 6
# Abhigyaan Deep

import math
import numpy as np
import os
import matplotlib.pyplot as plt
import concurrent.futures

"""
got sick of jumping between workspaces so now just place airfoil.dat adjacent
to this script :)
"""
airfoil_deg = np.loadtxt(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "airfoil.dat")
)
airfoil = airfoil_deg.copy()
airfoil[:, 0] = np.deg2rad(airfoil[:, 0])

N = 10  # sections
epsilon = 1e-10  # threshold
beta = 1e-3  # damping

V_infinity = 65

S = 17
AR = 8
lambda_ = 0.7
b = (S * AR) ** (1 / 2)
c_r = 2 * S / (b * (1 + lambda_))
c_t = lambda_ * c_r

alpha_0_deg = -17
alpha_1_deg = 17
d_alpha_deg = 1
alphas_deg = np.linspace(
    alpha_0_deg, alpha_1_deg, int((alpha_1_deg - alpha_0_deg) / d_alpha_deg) + 1
)
alphas = np.deg2rad(alphas_deg)

# incidence angles at root and tip
i_r = 1.5 * math.pi / 180
i_t = -1.5 * math.pi / 180

delta_z = b / N


def c_l(alpha):
    for i in range(len(airfoil) - 1):
        if airfoil[i][0] <= alpha <= airfoil[i + 1][0]:
            x0, y0 = airfoil[i]
            x1, y1 = airfoil[i + 1]
            return y0 + (alpha - x0) * (y1 - y0) / (x1 - x0)

    return None


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

        Gamma_new_i = V_infinity * c_i / 2 * C_l_i
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


# i just love the word ephemeral; here, it means initial
def Gamma_ephemeral(i):
    z_i = -b / 2 + delta_z / 2 + (i - 1) * delta_z
    return Gamma_0 * math.sqrt(1 - (z_i / (b / 2)) ** 2)


def converge(alpha):
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
        f"Converged alpha = {round(alpha * 180 / math.pi)}° in {iteration} iterations"
    )

    C_L = 2 * delta_z / (V_infinity * S) * sum(Gammas[i - 1] for i in range(1, N + 1))
    C_D_i = (
        -2
        * delta_z
        / (V_infinity * S)
        * sum(Gammas[i - 1] * math.sin(alpha_i_i(Gammas, i)) for i in range(1, N + 1))
    )

    return (C_L, C_D_i)


def main():
    C_Ls = []
    C_D_is = []

    # I threaded this because I am kinda tired of watching paint dry
    print(f"Spawning {os.cpu_count()} asynchronous compute threads...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(converge, alphas)

    for C_L, C_D_i in results:
        C_Ls.append(C_L)
        C_D_is.append(C_D_i)

    _, ax1 = plt.subplots()

    ax1.plot(
        airfoil_deg[:, 0],
        airfoil_deg[:, 1],
        color="tab:orange",
        label="C_l",
    )
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(alphas_deg, C_D_is, color="tab:red", label="C_D_i")
    ax2.set_ylabel("C_D_i", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.plot(alphas_deg, C_Ls, color="tab:blue", label="C_L")
    ax1.set_xlabel("Alpha (degrees)")
    ax1.set_ylabel("C_L", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("C_L and C_D_i vs Alpha")
    plt.show()


if __name__ == "__main__":
    main()
