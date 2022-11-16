
max_omp: max_cuda_openmp.cu
	nvcc --compiler-options -fopenmp -o max_cuda_openmp.exec max_cuda_openmp.cu

run_omp: max_omp
	srun -n 1 --gres=gpu:1 ./max_cuda_openmp.exec

max_mpi:
	nvcc -o max_cuda_mpi.exec --compiler-bindir mpic++ max_cuda_mpi.cu

run_mpi: max_mpi
	srun -n 1 --gres=gpu:1 ./max_cuda_mpi.exec

clean:
	rm -f *.exec