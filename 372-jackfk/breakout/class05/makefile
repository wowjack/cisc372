
%: %.c
	mpicc -o $@ $<
	srun -n 5 ./$@

clean:
	rm -rf hi10 addints
