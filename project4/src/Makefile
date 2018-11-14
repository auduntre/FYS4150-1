# Used as: make ising_model
# or: make ising_para
# typing "make" in terminal compiles both (same as: make all)

IDIR = ../include
ODIR = build

CC = mpic++
COMPILER = g++

CFLAGS1 = -O3 -c
CFLAGS2 = -I$(IDIR) -std=c++11 -Wall -larmadillo -O3

_DEPS = ising_lattice.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = ising_lattice.o ising_model.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cpp $(DEPS)
		$(COMPILER) -c -o $@ $< $(CFLAGS2)

all: para_ising ising_model

para_ising: para_ising.o
		$(CC) -O3 -o para_ising para_ising.cpp

para_ising.o:	para_ising.cpp
		$(CC) $(CFLAGS1) para_ising.cpp

ising_model: $(OBJ)
		$(COMPILER) $(CFLAGS1) $@ $<

clean:
		rm -rf *o ising_model para_ising 
