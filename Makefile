#!/bin/bash
INCLUDEPATH = -I/usr/local/include/ -I/usr/include -I/opt/OpenBLAS/include
LIBRARYPATH = -L/usr/local/lib -L/opt/OpenBLAS/lib
LIBRARY = -lpthread -lopenblas -lm -lgflags -fopenmp 
CPP_tag = -std=gnu++11 -fopenmp

LIB=/home/services/xiaoshu/lib
INCLUDE=/home/services/xiaoshu/include

all:ffm_ftrl_mpi rm

ffm_ftrl_mpi:main.o
	mpicxx $(CPP_tag) -g -o ffm_ftrl_mpi main.o $(LIBRARYPATH) $(LIBRARY)

main.o: src/main.cpp 
	mpicxx $(CPP_tag) $(INCLUDEPATH) -c src/main.cpp -DGLFAGS_NAMESPACE=google 
rm:
	rm main.o

clean:
	rm -f *~ ffm_ftrl_mpi predict *.o
