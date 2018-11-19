"""Program for plotting energy results obtained from para_ising."""

import matplotlib.pyplot as plt
import numpy as np


lattice = ["40", "60", "80", "100"]
cycles = 10000
temp = '2.000000'


for latt in lattice:
    cyc = str(cycles)
    lat = str(latt)
    inittemp = str(temp)
    x = np.loadtxt("results/mpi/datafiles/Lattice" + "_" + lat + "_" + cyc + "_" + inittemp + ".dat", skiprows=1)


    #plt.figure(figsize=(12,8))
    plt.title("Energy")
    plt.xlabel("T")
    plt.ylabel("<E>/L²")

    plt.grid(True)

    plt.plot(x[:, 0], x[:, 1], 'r-', linewidth = 4, label = 'L = ' + lat)

    plt.legend(loc='upper left', fontsize = 14)
    plt.savefig("results/mpi/plots/prob_e/Energy_" + lat + "_" + cyc + "_" + inittemp + ".png")
    #plt.show()
    plt.close() # Avoids overlaying plots


for latt in lattice:
    cyc = str(cycles)
    lat = str(latt)
    inittemp = str(temp)

    x = np.loadtxt("results/mpi/datafiles/Lattice" + "_" + lat + "_" + cyc + "_" + inittemp + ".dat", skiprows=1)

    #plt.figure(figsize=(12,8))
    plt.title("Magnetization")
    plt.xlabel("T")
    plt.ylabel("<|M|>/L²")

    plt.grid(True)

    plt.plot(x[:, 0], x[:, 3], 'b-', linewidth = 4, label = 'L = ' + lat)

    plt.legend(loc='lower left', fontsize = 14)
    plt.savefig("results/mpi/plots/prob_e/Magnetization_" + lat + "_" + cyc + "_" + inittemp + ".png")
    #plt.show()
    plt.close()

for lat in lattice:
    cyc = str(cycles)
    lat = str(lat)
    inittemp = str(temp)

    x = np.loadtxt("results/mpi/datafiles/Lattice" + "_" + lat + "_" + cyc + "_" + inittemp + ".dat", skiprows=1)

    #plt.figure(figsize=(12,8))
    plt.title("Absolute Magnetization")
    plt.xlabel("T")
    plt.ylabel("<|M|>/L²")

    plt.grid(True)

    plt.plot(x[:, 0], x[:, 5], 'c-', linewidth = 4, label = 'L = ' + lat)

    plt.legend(loc='lower left', fontsize = 14)
    plt.savefig("results/mpi/plots/prob_e/Abs_Magnetization_" + lat + "_" + cyc + "_" + inittemp + ".png")
    #plt.show()
    plt.close()


for latt in lattice:
    cyc = str(cycles)
    lat = str(latt)
    inittemp = str(temp)

    x = np.loadtxt("results/mpi/datafiles/Lattice" + "_" + lat + "_" + cyc + "_" + inittemp + ".dat", skiprows=1)

    #plt.figure(figsize=(12,8))
    plt.title("Susceptibility")
    plt.xlabel("T")
    plt.ylabel("chi/L²")

    plt.grid(True)

    plt.plot(x[:, 0], x[:, 4], 'g-', linewidth = 4, label = 'L = ' + lat)

    plt.legend(loc='upper left', fontsize = 14)
    plt.savefig("results/mpi/plots/prob_e/Susceptibility_" + lat + "_" + cyc + "_" + inittemp + ".png")
    #plt.show()
    plt.close()

for latt in lattice:
    cyc = str(cycles)
    lat = str(latt)
    inittemp = str(temp)

    x = np.loadtxt("results/mpi/datafiles/Lattice" + "_" + lat + "_" + cyc + "_" + inittemp + ".dat", skiprows=1)

    #plt.figure(figsize=(12,8))
    plt.title("Specific Heat")
    plt.xlabel("T")
    plt.ylabel("C_V/L²")

    plt.grid(True)

    plt.plot(x[:, 0], x[:, 2], 'k-', linewidth = 4, label = 'L = ' + lat)

    plt.legend(loc='upper left', fontsize = 14)
    plt.savefig("results/mpi/plots/prob_e/Specific_Heat_" + lat + "_" + cyc + "_" + inittemp + ".png")
    #plt.show()
    plt.close()
