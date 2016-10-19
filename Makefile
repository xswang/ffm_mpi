#!/bin/bash
INCLUDEPATH = -I/usr/local/include/ -I/usr/include -I/opt/OpenBLAS/include
LIBRARYPATH = -L/usr/local/lib -L/opt/OpenBLAS/lib
LIBRARY = -lpthread -lopenblas -lm -lgflags 
CPP_tag = -std=gnu++11

LIB=/home/services/xiaoshu/lib
INCLUDE=/home/services/xiaoshu/include

all:ffm_ftrl_mpi

ffm_ftrl_mpi:main.o
	mpicxx $(CPP_tag) -o ffm_ftrl_mpi main.o $(LIBRARYPATH) $(LIBRARY)

main.o: src/main.cpp 
	mpicxx $(CPP_tag) $(INCLUDEPATH) -c src/main.cpp -DGLFAGS_NAMESPACE=google 

clean:
	rm -f *~ ffm_ftrl_mpi predict *.o
