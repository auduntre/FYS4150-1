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
/*#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"*/
#include <mpi.h>

#include "ising_lattice.h"

using namespace  std;
using namespace arma;


int main (int argc, char* argv[])
{
    int getStartStep (int NTempSteps, int NProcesses, int RankProcess);

    string filename;
    int NSpins, NTempSteps, NProcesses, RankProcess, MonteCarloCycles;
    double InitialTemp, FinalTemp, TempStep;

    //  MPI initializations
    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &NProcesses);
    MPI_Comm_rank (MPI_COMM_WORLD, &RankProcess);
    
    if (RankProcess == 0 && argc <= 5) {
        cout << "Bad Usage: " << argv[0] <<
        " read output file, Number of spins, MC cycles, initial and final temperature and tempurate step" << endl;
        exit(1);
    } 
    else if ((RankProcess == 0) && (argc > 1)) {
        filename = argv[1];
        NSpins = atoi(argv[2]);
        MonteCarloCycles = atoi(argv[3]);
        InitialTemp = atof(argv[4]);
        FinalTemp = atof(argv[5]);
        TempStep = atof(argv[6]);

        // Temperature must be increasing
        if (FinalTemp < InitialTemp + TempStep) {
            cout << "Final temperature cannot be lower than initial tempearture"
                 << " + temperature step" << endl;
            exit(1);
        }
        
        NTempSteps = 1;
        for (double Temp = InitialTemp + TempStep; Temp <= FinalTemp; Temp += TempStep) {
            NTempSteps++;
        }
    }

    // broadcast to all nodes common variables since only master node reads from command line
    MPI_Bcast (&MonteCarloCycles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&NSpins, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&InitialTemp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&FinalTemp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&TempStep, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&NTempSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Step dividers to divide temp between processes
    int startStep = getStartStep(NTempSteps, NProcesses, RankProcess);
    int endStep = NTempSteps - 1;  // Last end step must be at the end
    int commStep;

    // Sending start step to process rank - 1  
    if (RankProcess > 0 && NProcesses > 1) {
        commStep = startStep;
        MPI_Send (&commStep, 1, MPI_INT, RankProcess - 1, 0, MPI_COMM_WORLD);
    }

    // Recieve proces (rank + 1)'s start step to set own end step 
    if (RankProcess < NProcesses - 1 && NProcesses > 1) {
        MPI_Recv (&commStep, 1, MPI_INT, RankProcess + 1, 0, 
                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        endStep = commStep - 1;
    }

    vec temps = linspace<vec>(InitialTemp, FinalTemp, NTempSteps);
    mat totalExpectationValues(5, NTempSteps, fill::zeros);
    mat localExpectationValues(5, NTempSteps, fill::zeros);
    
    // Informing of temperature step
    if (RankProcess == 0 && (temps[1] - temps[0] != TempStep)) {
        cout << "Temperature step set to: " << temps[1] - temps[0] << endl;
    }

    // Start Monte Carlo sampling by looping over the selected Temperatures
    double TimeStart, TimeEnd, TotalTime;
    TimeStart = MPI_Wtime ();

    for (int step = startStep; step <= endStep; step++) {
        vec tmpEV = zeros<vec>(5);

        // Start Monte Carlo computation and get local expectation values
        MetropolisSampling(NSpins, MonteCarloCycles, temps[step], tmpEV);
        localExpectationValues.col(step) = tmpEV;
    }

    // Send all values to process 0
    for (int ij = 0; ij < 5 * NTempSteps; ij++) {
        MPI_Reduce (&localExpectationValues(ij), &totalExpectationValues(ij),
                    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // only master node opens and prints to file
    if (RankProcess == 0) {
        // Declare new file name and add lattice size to file name, 
        string fileout = filename;
        string argument = "_" + to_string(NSpins) + "_" + to_string(MonteCarloCycles);
        fileout.append(argument);
        
        ofstream ofile("/results/");
        ofile.open("results/mpi/" + fileout);

        // Print to file
        for (int step = 0; step < NTempSteps; step++) {
            WriteResultstoFile(ofile, NSpins, MonteCarloCycles, temps[step], 
                               totalExpectationValues.col(step));
        }

        ofile.close();  // close output file
    }
    
    TimeEnd = MPI_Wtime ();
    TotalTime = TimeEnd - TimeStart;
    
    if ( RankProcess == 0) {
        cout << "Time = " <<  TotalTime  << " on number of processors: "  << NProcesses  << endl;
    }
    
    MPI_Finalize ();
    return 0;
}


int getStartStep (int NTempSteps, int NProcesses, int RankProcess) 
{
    int rankSteps = NTempSteps / NProcesses;
    int reminderSteps = NTempSteps % NProcesses;
    int startStep;

    // Divinding Steps
    if (RankProcess <= reminderSteps) {
        startStep = (rankSteps + 1) * RankProcess;
    } else {
        startStep = (rankSteps + 1) * reminderSteps
                  + rankSteps * (RankProcess - reminderSteps); 
    }

    return startStep;
}