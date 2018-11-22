#!/usr/bin/env python
"""Program to plot 2 x 2 lattice case"""

import matplotlib.pyplot as plt
import numpy as np

N = 4 # number of spins
J = k = T = 1.0  # Scaled so that these are one
beta = 1.0 / (k * T)

# Partition function
Z  = 2 * np.exp(8 * J * beta) + 2 * np.exp(-8 * J * beta) + 12

# Energy
E  = - (J / Z) * 16 *  (np.exp(8 * J * beta) - np.exp(-8 * J * beta))
E2 = (J**2 / Z) * 128 * (np.exp(8 * J * beta) + np.exp(-8 * J * beta))

# Magnetization
M    = 0
Mabs = (1 / Z) * 8 * (np.exp(8 * J * beta) + 16)
M2   = (1 / Z) * 32 * (np.exp(8 * J * beta) + 1)


# Specific Heat
Cv = (E2 - E**2) / (k * T**2)
# Suceptibility
X = (M2 - M) * beta

# Divide by number of spins
anylytical_values = np.asarray([T * N, E, Cv, M, X, Mabs]) / N

MCC = [10**i for i in range(2, 8)] # Monte Carlo cycles

print(anylytical_values)
for mcc in MCC:
    x = np.loadtxt("../results/nopara/numtest_2_" + str(mcc))[0, :]
    print(x)