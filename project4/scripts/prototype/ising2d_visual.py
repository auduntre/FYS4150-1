# Taken from lecture notes in FYS3150
# visulize 2d ising model

from matplotlib import pyplot as plt
from ising2d import periodic
from numba import jit

import numpy as np
import sys, math
import pygame

screen = None
font = None
BLOCKSIZE = 10


@jit
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
    spin_matrix = np.zeros( (size,size), np.int8) + 1

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
        for s in range(size**2):
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

        if game:
            visualize(spin_matrix, temp, E / float(size**2), M / float(size**2), method) 

    E_av       /= float(trials)
    E2_av      /= float(trials)
    M_av       /= float(trials)
    M2_av      /= float(trials)
    Mabs_av    /= float(trials)
    #Calculate variance and normalize to per-point and temp
    E_variance  = (E2_av - E_av * E_av) / float(size * size * temp * temp)
    M_variance  = (M2_av - M_av * M_av) / float(size * size * temp)
    #Normalize returned averages to per-point
    E_av       /= float(size * size)
    M_av       /= float(size * size)
    Mabs_av    /= float(size * size)

    return (E_av, E_variance, M_av, M_variance, Mabs_av)


def visualize(spin_matrix, temp, E, M, method):
    """
    Visualize the spin matrix

    Methods:
    method = -1:No visualization (testing)
    method = 0: Just print it to the terminal
    method = 1: Pretty-print to terminal
    method = 2: SDL/pygame single-pixel
    method = 3: SDL/pygame rectangle
    """

    #Simple terminal dump
    if method == 0:
        print("temp:", temp, "E:", E, "M:", M)
        print(spin_matrix)
    #Pretty-print to terminal
    elif method == 1:
        out = ""
        size = len(spin_matrix)
        for y in range(size):
            for x in range(size):
                if spin_matrix[x, y] == 1:
                    out += "X"
                else:
                    out += " "
            out += "\n"

        print("temp:", temp, "E:", E, "M:", M)
        print(out + "\n")
    #SDL single-pixel (useful for large arrays)
    elif method == 2:
        size = len(spin_matrix)
        screen.lock()
        for y in range(size):
            for x in range(size):
                if spin_matrix[x, y] == 1:
                    screen.set_at((x,y), (0,0,255))
                else:
                    screen.set_at((x,y), (255,0,0))

        screen.unlock()
        pygame.display.flip()
    #SDL block (usefull for smaller arrays)
    elif method == 3:
        size = len(spin_matrix)
        screen.lock()
        for y in range(size):
            for x in range(size):
                if spin_matrix[x, y] == 1:
                    rect = pygame.Rect(x * BLOCKSIZE, y * BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
                    pygame.draw.rect(screen, (0, 0, 255), rect)
                else:
                    rect = pygame.Rect(x * BLOCKSIZE, y * BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
                    pygame.draw.rect(screen, (255, 0, 0), rect)
        screen.unlock()
        pygame.display.flip()
    #SDL block w/ data-display
    elif method == 4:
        size = len(spin_matrix)
        screen.lock()
        for y in xrange(size):
            for x in xrange(size):
                if spin_matrix[x,y] == 1:
                    rect = pygame.Rect(x * BLOCKSIZE, y * BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
                    pygame.draw.rect(screen, (255, 255, 255), rect)
                else:
                    rect = pygame.Rect(x * BLOCKSIZE, y * BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
                    pygame.draw.rect(screen, (0, 0, 0), rect)
        s = font.render("<E> = %5.3E; <M> = %5.3E" % E, M, False, (255,0,0))
        screen.blit(s, (0, 0))
        
        screen.unlock()
        pygame.display.flip()


def main_game():
    size   = 100
    trials = 10
    temp   = 2.1
    method = 3
    game = True

    pygame.init()
    #Initialize pygame
    if method == 2 or method == 3 or method == 4:
        pygame.init()
        
        global screen
        if method == 2:
            screen = pygame.display.set_mode((size, size))
        elif method == 3:
            screen = pygame.display.set_mode((size * 10, size * 10))
        elif method == 4:
            screen = pygame.display.set_mode((size * 10, size * 10))
            font   = pygame.font.Font(None, 12)

    (E_av, E_variance, M_av, M_variance, Mabs_av) = monteCarlo(temp, size, trials, game, method)
    print("%15.8E %15.8E %15.8E %15.8E %15.8E %15.8E\n" % (temp, E_av, E_variance, M_av, M_variance, Mabs_av))


    pygame.quit()

if __name__ == "__main__":
    main_game()
