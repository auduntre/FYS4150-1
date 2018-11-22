# Code for the two-dimensional Ising model with periodic boundary conditions
# taken from lecture notes in FYS3150 

from matplotlib import pyplot as plt
from numba import jit, prange

import numpy as np
import sys, math

@jit(nopython=True)
def periodic(i, limit, add):
    """
    Choose correct matrix index with periodic
    boundary conditions

    Input:
    - i:     Base index
    - limit: Highest \"legal\" index
    - add:   Number to add or subtract from i
    """
    return (i + limit + add) % limit


@jit(nopython=True)
def monteCarlo(temp, size, trials, game=False, method=1):
    """
    Calculate the enerInvalid usage of BoundFunction(array.item for array(int64, 2d, C)) with parameters (int64, int64)
    * parameterizegy and magnetization
    (\"straight\" and squared) for a given temperature

    Input:
    - temp:   Temperature to calculate for
    - size:   dimension of square matrix
    - trials: Monte-carlo trials (how many times do we
                                  flip the matrix?)

    Output:
    - E_av:       Energy of matrix averaged over trials, normalized to spins**2
    - E_variance: Variance of energy, same normalization * temp**2
    """

    #Setup spin matrix, initialize to ground state
    spin_matrix = np.zeros((size,size), np.int8) + 1

    #Create and initialize variables
    E = M = 0
    E_av = E2_av = M_av = M2_av = Mabs_av = 0

    #Setup array for possible energy changes
    w = np.zeros(17, np.float64)
    for de in range(-8, 9, 4): #include +8
        w[de+8] = math.exp(-de/temp)

    # Calculate initial magnetization
    M = spin_matrix.sum()
    #Calculate initial energy
    for j in range(size):
        for i in range(size):
            E -= spin_matrix[i, j] * \
                 (spin_matrix[periodic(i, size, -1), j] + spin_matrix[i, periodic(j, size, 1)])

    #Start metropolis MonteCarlo computation
    for i in range(trials):
        #Metropolis
        #Loop over all spins, pick a random spin each time
        for s in range(size * size):
            x = int(np.random.random() * size)
            y = int(np.random.random() * size)
            deltaE = 2 * spin_matrix[x,y] * \
                     (spin_matrix[periodic(x, size, -1), y] +\
                      spin_matrix[periodic(x, size, 1),  y] +\
                      spin_matrix[x, periodic(y, size, -1)] +\
                      spin_matrix[x, periodic(y, size, 1)])
            if np.random.random() <= w[deltaE+8]:
                #Accept!
                spin_matrix[x,y] *= -1
                M += 2 * spin_matrix[x, y]
                E += deltaE

        #Update expectation values
        E_av    += E
        E2_av   += E**2
        M_av    += M
        M2_av   += M**2
        Mabs_av += int(math.fabs(M))

    ftrials = float(trials)
    fsize2  = float(size * size)

    E_av       /= ftrials
    E2_av      /= ftrials
    M_av       /= ftrials
    M2_av      /= ftrials
    Mabs_av    /= ftrials
    #Calculate variance and normalize to per-point and temp
    E_variance  = (E2_av - E_av * E_av) / (fsize2 *  temp * temp)
    M_variance  = (M2_av - M_av * M_av) / (fsize2 * temp)
    #Normalize returned averages to per-point
    E_av       /= fsize2
    M_av       /= fsize2
    Mabs_av    /= fsize2

    return (E_av, E_variance, M_av, M_variance, Mabs_av)


@jit(nopython=True, parallel=True)
def temp_loop (temps, size, trials):
    """Loop over all the temps."""
    Dim = len(temps)
    energy = np.zeros(Dim)
    heatcapacity = np.zeros(Dim)
    temperature = np.zeros(Dim)
    magnetization = np.zeros(Dim)
    
    for i in prange(Dim):
        (E_av, E_variance, M_av, _, _) = monteCarlo(temps[i], size, trials)
        temperature[i] = temps[i]
        energy[i] = E_av
        heatcapacity[i] = E_variance
        magnetization[i] = M_av

    return (energy, heatcapacity, temperature, magnetization)


def main():
    """Main program"""

    # values of the lattice, number of Monte Carlo cycles and temperature domain
    size        = 20
    trials      = 100000
    temp_init   = 1.8
    temp_end    = 2.6
    temp_step   = 0.01


    temps = np.arange(temp_init, temp_end+temp_step/2, temp_step, float)
    (energy, heatcapacity, temperature, magnetization) = temp_loop(temps, size, trials)
    
    plt.figure(1)
    
    plt.subplot(311)
    plt.axis([1.8,2.6,-2.0, -1.0])
    plt.xlabel(r'Temperature $J/(k_B)$')
    plt.ylabel(r'Average energy per spin  $E/N$')
    plt.plot(temperature, energy, 'b-')
    
    plt.subplot(312)
    plt.axis([1.8,2.6, 0.0, 2.0])
    plt.plot(temperature, heatcapacity, 'r-')
    plt.xlabel(r'Temperature $J/(k_B)$')
    plt.ylabel(r'Heat capacity per spin  $C_V/N$')

    plt.subplot(313)
    plt.axis([1.8,2.6, 0.0, 2.0])
    plt.plot(temperature, magnetization, 'g-')
    plt.xlabel(r'Temperature $J/(k_B)$')
    plt.ylabel(r'Magnetization per spin  $M/N$')
    plt.savefig('energycv.pdf')
    #plt.show()


if __name__ == "__main__":
    main()

