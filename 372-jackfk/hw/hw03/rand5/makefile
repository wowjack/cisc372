
rand5.exec: rand5.c
	mpicc -o rand5.exec rand5.c

test: rand5.exec
	srun -n 5 ./rand5.exec

clean:
	rm -rf *.exec