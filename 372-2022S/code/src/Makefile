
all:
	$(MAKE) -C anim all
	$(MAKE) -C seq all
	$(MAKE) -C mpi all
	$(MAKE) -C omp all
	$(MAKE) -C pthread all
#ifneq ($(NVCC),)
#	$(MAKE) -C cuda all
#endif

test:
	$(MAKE) -C anim test
	$(MAKE) -C seq test
	$(MAKE) -C mpi test
	$(MAKE) -C omp test
	$(MAKE) -C pthread test
#ifneq ($(NVCC),)
#	$(MAKE) -C cuda test
#endif

clean:
	rm -f a.out *~ *.tmp *.o *.exec
	$(MAKE) -C anim clean
	$(MAKE) -C seq clean
	$(MAKE) -C mpi clean
	$(MAKE) -C omp clean
	$(MAKE) -C pthread clean
#	$(MAKE) -C cuda clean

.PHONY: all test clean
