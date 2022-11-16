
%: %.c
	mpicc -o out $@.c
	srun -n 10 ./out
	rm out