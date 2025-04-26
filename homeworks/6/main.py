# AERE 310 HW 6
# Abhigyaan Deep

import math
import numpy
import pint

ur = pint.UnitRegistry()

airfoil = numpy.loadtxt("homeworks/6/airfoil.dat")


# alpha in deg
def c_l(alpha):
    for i in range(len(airfoil) - 1):
        if airfoil[i][0] <= alpha <= airfoil[i + 1][0]:
            # Linear interpolation formula
            x0, y0 = airfoil[i]
            x1, y1 = airfoil[i + 1]
            return y0 + (alpha - x0) * (y1 - y0) / (x1 - x0)
    return None


N = 10

V_infinity = 65 * ur.m / ur.s

S = 17 * ur.m**2
AR = 8
lambda_ = 0.7
b = (S * AR) ** (1 / 2)
c_r = 2 * S / (b * (1 + lambda_))
c_t = lambda_ * c_r

# incidence angles at root and tip
i_r = (1.5 * ur.deg).to(ur.rad).magnitude
i_t = (-1.5 * ur.deg).to(ur.rad).magnitude

# compute domain
alpha_0 = -17 * ur.deg
alpha_1 = 17 * ur.deg

delta_z = b / N

c_l_z_0 = c_l(0)
Gamma_0 = (V_infinity * c_r * c_l_z_0) / 2

for i in range(1, N + 1):
    z_i = -b / 2 + delta_z / 2 + (i - 1) * delta_z
    Gamma_i_0 = Gamma_0 * (1 - (z_i / (b / 2)) ** 2) ** (1 / 2)

    # alpha_i_i = (-1 / (4 * math.pi * V_infinity * delta_z)) * sum(
    #     Gamma(j) * D(i - j) for j in range(1, N + 1)
    # )

    # print(alpha_i_i)


def Gamma(i):
    return 0


def D(x):
    return 4 / (1 - 4 * x**2)


C_L = ((2 * delta_z) / (V_infinity * S)) * sum(Gamma(i) for i in range(1, N + 1))
