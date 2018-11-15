#ifndef ISING_LATTICE_H
#define ISING_LATTICE_H

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <random>
#include <armadillo>
#include <string>

using namespace std;
using namespace arma;

// inline function for PeriodicBoundary boundary conditions
inline int PeriodicBoundary(int i, int limit, int add);

// Function to initialise energy and magnetization
void InitializeLattice(int, mat &, double&, double&);

// The metropolis algorithm including the loop over Monte Carlo cycles
void MetropolisSampling(int NSpins, int MCcycles, double Temperature, vec &ExpectationValues);

// prints to file the results of the calculations
void WriteResultstoFile(ofstream &ofile, int NSpins, int MonteCarloCycles, double temperature, vec ExpectationValues);

#endif
