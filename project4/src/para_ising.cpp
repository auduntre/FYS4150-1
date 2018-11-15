/*
   Program to solve the two-dimensional Ising model
   with zero external field and no parallelization
   Parallel version using MPI
   The coupling constant J is set to J = 1
   Boltzmann's constant = 1, temperature has thus dimension energy
   Metropolis aolgorithm  is used as well as periodic boundary conditions.
   The code needs an output file on the command line and the variables mcs, nspins,
   initial temp, final temp and temp step.
   Run as
   mpiexec -np NumProcessors ./executable Outputfile NSpins, MCcycles, InitialTemp, FinalTemp, and TempStep.
   ./test.x Lattice 100 10000000 2.1 2.4 0.01
   Compile and link can be done with Makefile or by
   Compile and link as
   c++ -O3 -std=c++11 -Rpass=loop-vectorize -o Ising.x IsingModel.cpp -larmadillo
*/

// Remember to make sure to have the right directory for YOUR system
#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
#include "ising_lattice.h"

using namespace  std;
using namespace arma;

// output file
ofstream ofile("/results/");

// Main program begins here
int main (int argc, char* argv[])
{
    string filename;
    int NSpins, MonteCarloCycles;
    double InitialTemp, FinalTemp, TempStep;
    int NProcesses, RankProcess;
  
    //  MPI initializations
    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &NProcesses);
    MPI_Comm_rank (MPI_COMM_WORLD, &RankProcess);
    
    if (RankProcess == 0 && argc <= 5) {
        cout << "Bad Usage: " << argv[0] <<
        " read output file, Number of spins, MC cycles, initial and final temperature and tempurate step" << endl;
        exit(1);
    }
    
    if ((RankProcess == 0) && (argc > 1)) {
        filename=argv[1];
        NSpins = atoi(argv[2]);
        MonteCarloCycles = atoi(argv[3]);
        InitialTemp = atof(argv[4]);
        FinalTemp = atof(argv[5]);
        TempStep = atof(argv[6]);
    }
    
    // Declare new file name and add lattice size to file name, only master node opens file
    if (RankProcess == 0) {
        string fileout = filename;
        string argument = "_" + to_string(NSpins) + "_" + to_string(MonteCarloCycles);
        fileout.append(argument);
        ofile.open("results/mpi/" + fileout);
    }
  
    // broadcast to all nodes common variables since only master node reads from command line
    MPI_Bcast (&MonteCarloCycles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&NSpins, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&InitialTemp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&FinalTemp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&TempStep, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Start Monte Carlo sampling by looping over the selected Temperatures
    double  TimeStart, TimeEnd, TotalTime;
    TimeStart = MPI_Wtime();
    
    for (double Temperature = InitialTemp; Temperature <= FinalTemp; Temperature+=TempStep){
        vec LocalExpectationValues = zeros<mat>(5);
        
        // Start Monte Carlo computation and get local expectation values
        MetropolisSampling(NSpins, MonteCarloCycles, Temperature, LocalExpectationValues);
        
        // Find total average
        vec TotalExpectationValues = zeros<mat>(5);
        
        for( int i =0; i < 5; i++){
            MPI_Reduce(&LocalExpectationValues[i], &TotalExpectationValues[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        
        if ( RankProcess == 0) {
            WriteResultstoFile(ofile, NSpins, MonteCarloCycles*NProcesses, Temperature, TotalExpectationValues);
        }
    }
    
    if(RankProcess == 0)  ofile.close();  // close output file
    
    TimeEnd = MPI_Wtime();
    TotalTime = TimeEnd-TimeStart;
    
    if ( RankProcess == 0) {
        cout << "Time = " <<  TotalTime  << " on number of processors: "  << NProcesses  << endl;
    }
    
    // End MPI
    MPI_Finalize ();
    return 0;
}