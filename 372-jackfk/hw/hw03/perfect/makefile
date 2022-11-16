
all: perfect_mpi.c
	mpicc -o perfect_mpi.exec perfect_mpi.c

run: all
	srun -n 10 ./perfect_mpi.exec 50000 

clean:
	rm -rf *.exec *.o
