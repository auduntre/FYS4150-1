#!/usr/bin/env python
"""Program for plotting results obtained from ising_model. """

import matplotlib.pyplot as plt
import numpy as np

filename = "../results/mpi/bigL_"
MCC = 200000
strmcc = "_" + str(MCC)
Lvalues = [40, 60, 80, 100]

Ts = np.loadtxt(filename + "40" + strmcc)[:, 0]
values = ["$<E>$", "$<|M|>$", "$C_v$", "$X$"]
loadvalues = [1, 5, 2, 4] # Where to find the valus in the files

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 14})

for vi, value in zip(loadvalues, values):
    plt.grid(True)

    for i, L in enumerate(Lvalues):
        v = np.loadtxt(filename + str(L) + strmcc)[:, vi] #load values   
        plt.plot(Ts, v, label="L={}".format(L))

    plt.xlabel("Temperature (kT/J)")
    plt.title(r"{}/N for T $\in$ [{},{}] MCC = {} ".format(value, Ts[0], Ts[-1], MCC))
    plt.legend()
    plt.show()
    plt.close()
