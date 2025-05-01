# AERE 310 HW 6
# Abhigyaan Deep

import math
import numpy
import os
import pint

ur = pint.UnitRegistry()

"""
got sick of jumping between workspaces so now just place airfoil.dat adjacent
to this script :)
"""
airfoil = numpy.loadtxt(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "airfoil.dat")
)


def c_l(_alpha):
    alpha = _alpha.to(ur.deg).magnitude

    for i in range(len(airfoil) - 1):
        if airfoil[i][0] <= alpha <= airfoil[i + 1][0]:
            x0, y0 = airfoil[i]
            x1, y1 = airfoil[i + 1]
            return y0 + (alpha - x0) * (y1 - y0) / (x1 - x0)

    return None


N = 10  # sections
M = int(1e3)  # iterations
beta = 1e-2  # damping

V_infinity = 65 * ur.m / ur.s

S = 17 * ur.m**2
AR = 8
lambda_ = 0.7
b = (S * AR) ** (1 / 2)
c_r = 2 * S / (b * (1 + lambda_))
c_t = lambda_ * c_r


def c(z):
    x = (abs(z) / (b / 2)).to_base_units().magnitude
    return c_r * (1 - x) + c_t * x


# incidence angles at root and tip
i_r = (1.5 * ur.deg).to(ur.rad).magnitude
i_t = (-1.5 * ur.deg).to(ur.rad).magnitude


def i(z):
    x = (abs(z) / (b / 2)).to_base_units().magnitude
    return i_r * (1 - x) + i_t * x


# compute domain
alpha_0 = -17 * ur.deg
alpha_1 = 17 * ur.deg

delta_z = b / N

c_l_z_0 = c_l(i(0) * ur.rad)  # c_l of the incidence angle at z = 0
Gamma_0 = (V_infinity * c_r * c_l_z_0) / 2


# i just love the word ephemeral; here, it means initial
def Gamma_ephemeral(i):
    z_i = -b / 2 + delta_z / 2 + (i - 1) * delta_z
    return Gamma_0 * (1 - (z_i / (b / 2)) ** 2) ** (1 / 2)


Gammas = [Gamma_ephemeral(i) for i in range(1, N + 1)]


def D(x):
    return 4 / (1 - 4 * x**2)


# 0 for testing
alpha_i = 0


def iterate():
    Gammas_new = []

    for i in range(1, N + 1):
        alpha_i_i = (-1 / (4 * math.pi * V_infinity * delta_z)) * sum(
            Gammas[j - 1] * D(i - j) for j in range(1, N + 1)
        )
        alpha_eff_i = alpha_i + alpha_i_i

        z_i = -b / 2 + (i - 0.5) * delta_z
        c_i = c(z_i)

        C_l_i = c_l(alpha_eff_i * ur.rad)

        Gamma_new_i = V_infinity * c_i * C_l_i
        Gamma_old_i = Gammas[i - 1]
        Gamma = Gamma_old_i + beta * (Gamma_new_i - Gamma_old_i)

        Gammas_new.append(Gamma)

    return Gammas_new


print("ephemeral values:")
for Gamma in Gammas:
    print(Gamma)

for i in range(M):
    Gammas = iterate()

print("converged values:")
for Gamma in Gammas:
    print(Gamma)
