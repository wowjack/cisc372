
all: build
	srun -n 1 ./out > pi_strong.dat
	srun -n 2 ./out >> pi_strong.dat
	srun -n 4 ./out >> pi_strong.dat
	srun -n 6 ./out >> pi_strong.dat
	srun -n 8 ./out >> pi_strong.dat
	srun -n 10 ./out >> pi_strong.dat

build:
	mpicc -o out pi_rect_mpi.c

clean:
	rm -rf out *.dat
