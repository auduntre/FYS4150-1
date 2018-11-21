# Used as: make ising_model
# or: make ising_para
# typing "make" in terminal compiles both (same as: make all)

IDIR = ../include
ODIR = ../build
MPIODIR = $(ODIR)/mpi

CC = mpic++
COMPILER = g++

CFLAGS = -I$(IDIR) -std=c++11 -Wall -larmadillo -O3
CLINKFLAGS = -Wall -O3

_DEPS = ising_lattice.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = ising_lattice.o ising_model.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_MPIOBJ = ising_lattice.o para_ising.o 
MPIOBJ = $(patsubst %,$(MPIODIR)/%,$(_MPIOBJ))


$(ODIR)/%.o: %.cpp $(DEPS)
		$(COMPILER) -c -o $@ $< $(CFLAGS)

$(MPIODIR)/%.o: %.cpp $(DEPS)
		$(CC) -c -o $@ $< $(CFLAGS)


all: para_ising ising_model

para_ising: $(MPIOBJ)
		$(CC) -o $@ $^ $(CLINKFLAGS)

ising_model: $(OBJ) 
		$(COMPILER) -o $@ $^ $(CLINKFLAGS) 


clean:
		rm -rf $(ODIR)/*.o $(MPIODIR)/*.o ising_model para_ising 
