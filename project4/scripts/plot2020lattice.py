#!/usr/bin/env python
"""Program for plotting results obtained from ising_model"""


import matplotlib.pyplot as plt
import numpy as np

MCCE4 = [10000*i for i in range(1, 11)]
MCCE5 = [10*i for i in MCCE4]
MCCs = MCCE4 + MCCE5
Es = np.zeros(len(MCCs))
Ms = np.zeros(len(MCCs))

temps = [1.0, 2.4]

plt.rcParams.update({'font.size': 14})

for i, temp in enumerate(temps):
    for j, mcc in enumerate(MCCs):
        x = np.loadtxt("../results/mpi/Lattice_20_" + str(mcc))[i, :]

        Es[j] = x[1]
        Ms[j] = x[-1]

    plt.figure(figsize=(12,5))
    plt.title("Energy plot for temperature =  {} (kT/J)".format(temp))
    plt.plot(MCCs, Es)
    plt.ylabel("E/N")
    plt.xlabel("MCCs")
    plt.show()

    plt.figure(figsize=(12,5))
    plt.title("Magnitization plot for temperature =  {} (kT/J)".format(temp))
    plt.plot(MCCs, Ms)
    plt.ylabel("|M|/N")
    plt.xlabel("MCCs")
    plt.show()
