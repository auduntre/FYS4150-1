/*
    Program to solve the two-dimensional Ising model
    with zero external field and no parallelization.
    The coupling constant J is set to J = 1.
    Boltzmann's constant = 1, temperature has thus dimension energy.
    Metropolis algorithm is used as well as periodic boundary conditions.
    The code needs an output file on the command line and the variables
    NSpins, MCcycles, InitialTemp, FinalTemp, and TempStep.
*/

// Include
#include "ising_lattice.h"

using namespace std;
using namespace arma;

// output file
ofstream ofile("/results/");

int main(int argc, char* argv[])
{
    string filename;
    int NSpins, MCcycles;
    double InitialTemp, FinalTemp, TempStep;
    
    if (argc <= 5) {
        cout << "Bad Usage: " << argv[0] <<
        " read output file, Number of spins, MC cycles, intial and final temperature and temperature step" << endl;
        exit(1);
    }

    if (argc > 1) {
        filename = argv[1];
        NSpins = atoi(argv[2]);
        MCcycles = atoi(argv[3]);
        InitialTemp = atof(argv[4]);
        FinalTemp = atof(argv[5]);
        TempStep = atof(argv[6]);
    }

    // Declare new file name and add lattice size to file name
    string fileout = filename;
    string argument = "_" + to_string(NSpins) + "_" + to_string(MCcycles);
    fileout.append(argument);
    ofile.open("results/nopara/" + fileout);
    
    // Start Monte Carlo sampling by looping over the selected Temperatures
    for (double Temperature = InitialTemp; Temperature <= FinalTemp; Temperature += TempStep) {
        vec ExpectationValues = zeros<mat>(5);
        
        // Start Monte Carlo computation and get expectation values
        MetropolisSampling(NSpins, MCcycles, Temperature, ExpectationValues);
        
        //
        WriteResultstoFile(NSpins, MCcycles, Temperature, ExpectationValues);
    }
    ofile.close(); // close output file
    return 0;
    // end of main program
}

