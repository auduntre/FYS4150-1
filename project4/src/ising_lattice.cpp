#include "ising_lattice.h"

using namespace std;
using namespace arma;


inline int PeriodicBoundary (int i, int limit, int add) 
{
    return (i+limit+add) % (limit);
}


// function to initialise energy, spin matrix and magnetization
void InitializeLattice (int NSpins, mat &SpinMatrix,  double& Energy, double& MagneticMoment)
{
  // setup spin matrix and initial magnetization
    for(int x =0; x < NSpins; x++) {
        for (int y= 0; y < NSpins; y++){
            SpinMatrix(x,y) = 1.0; // spin orientation for the ground state
            MagneticMoment +=  (double) SpinMatrix(x, y);
        }
    }
    
    // setup initial energy
    for(int x =0; x < NSpins; x++) {
        for (int y= 0; y < NSpins; y++){
            Energy -=  (double) SpinMatrix(x, y)
    	               * (SpinMatrix(PeriodicBoundary(x, NSpins, -1), y)
    	                  + SpinMatrix(x, PeriodicBoundary(y, NSpins, -1)));
        }
    }
}


// The Monte Carlo part with the Metropolis algo with sweeps over the lattice
void MetropolisSampling (int NSpins, int MCcycles, double Temperature, vec &ExpectationValues)
{
    // Initialize the seed and call the Mersienne algo
    std::random_device rd;
    std::mt19937_64 gen(rd());
    
    // Set up the uniform distribution for x \in [0, 1]
    std::uniform_real_distribution<double> RandomNumberGenerator(0.0, 1.0);
    
    // Initialize the lattice spin values
    mat SpinMatrix = zeros<mat>(NSpins, NSpins);
    
    //    initialize energy and magnetization
    double Energy = 0.0;     
    double MagneticMoment = 0.0;
    
    // initialize array for expectation values
    InitializeLattice(NSpins, SpinMatrix, Energy, MagneticMoment);
    
    // setup array for possible energy changes
    vec EnergyDifference = zeros<mat>(17);
    for (int de =-8; de <= 8; de+=4) EnergyDifference(de+8) = exp(-de / Temperature);
    
    // Start Monte Carlo cycles
    for (int cycles = 1; cycles <= MCcycles; cycles++){
        // The sweep over the lattice, looping over all spin sites
        for (int xy = 0; xy < NSpins * NSpins; xy ++) {
            int ix = (int) (RandomNumberGenerator(gen) * (double) NSpins);
            int iy = (int) (RandomNumberGenerator(gen) * (double) NSpins);
            int deltaE = 2 * SpinMatrix(ix, iy)
                         * (SpinMatrix(ix, PeriodicBoundary(iy, NSpins, -1))
                            + SpinMatrix(PeriodicBoundary(ix, NSpins, -1), iy)
            	            + SpinMatrix(ix, PeriodicBoundary(iy, NSpins, 1))
            	            + SpinMatrix(PeriodicBoundary(ix, NSpins, 1), iy));

            if ( RandomNumberGenerator(gen) <= EnergyDifference(deltaE + 8) ) {
              	SpinMatrix(ix,iy) *= -1.0;  // flip one spin and accept new spin config
              	MagneticMoment += (double) (2.0 * SpinMatrix(ix, iy));
              	Energy += (double) deltaE;
            }
        }
        
        // update expectation values  for local node
        ExpectationValues(0) += Energy;    
        ExpectationValues(1) += Energy * Energy;
        ExpectationValues(2) += MagneticMoment;
        ExpectationValues(3) += MagneticMoment * MagneticMoment;
        ExpectationValues(4) += fabs(MagneticMoment);
    }
}


void WriteResultstoFile (ofstream &ofile, int NSpins, int MonteCarloCycles, double temperature, vec ExpectationValues)
{
    double norm = 1.0 / ((double) (MonteCarloCycles));  // divided by  number of cycles
    double E_ExpectationValues = ExpectationValues(0) * norm;
    double E2_ExpectationValues = ExpectationValues(1) * norm;
    double M_ExpectationValues = ExpectationValues(2) * norm;
    double M2_ExpectationValues = ExpectationValues(3) * norm;
    double Mabs_ExpectationValues = ExpectationValues(4) * norm;
    
    // all expectation values are per spin, divide by 1/NSpins/NSpins
    double AllSpins = 1.0 / ((double) (NSpins * NSpins));
    double HeatCapacity = (E2_ExpectationValues - E_ExpectationValues * E_ExpectationValues) * AllSpins / temperature / temperature;
    double MagneticSusceptibility = (M2_ExpectationValues - M_ExpectationValues * M_ExpectationValues) * AllSpins / temperature;
    
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << temperature;
    ofile << setw(15) << setprecision(8) << E_ExpectationValues * AllSpins;
    ofile << setw(15) << setprecision(8) << HeatCapacity;
    ofile << setw(15) << setprecision(8) << M_ExpectationValues * AllSpins;
    ofile << setw(15) << setprecision(8) << MagneticSusceptibility;
    ofile << setw(15) << setprecision(8) << Mabs_ExpectationValues * AllSpins << endl;
}