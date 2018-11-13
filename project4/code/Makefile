# Used as: make ising_model
# or: make ising_para
# typing "make" in terminal compiles both (same as: make all)


CC = mpic++
COMPILER = g++

CFLAGS1 = -O3 -c
CFLAGS2 = -c -Wall -larmadillo -O3

all: para_ising ising_model

para_ising: para_ising.o
					$(CC) -O3 -o para_ising para_ising.cpp


para_ising.o:	para_ising.cpp
					$(CC) $(CFLAGS1) para_ising.cpp



ising_model: ising_model.o
					$(COMPILER) -O3 -o ising_model ising_model.cpp

ising_model.o: ising_model.cpp
					$(COMPILER) $(CFLAGS2) ising_model.cpp

clean:
					rm -rf *o ising_model para_ising 
