#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <assert.h>

__global__ void kernel(int rank) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  printf("Hello from block %d, thread %d of the GPU, called by process %d\n",
	 bid, tid, rank);
}

int main (int argc, char * argv[]) {
  int rank, nprocs, required = MPI_THREAD_FUNNELED, provided;

  MPI_Init_thread(&argc, &argv, required, &provided);
  assert(provided == MPI_THREAD_FUNNELED);
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  kernel<<<3,4>>>(rank); // 3 blocks, 4 threads per block
#pragma omp parallel shared(rank,nprocs)
  {
    int tid = omp_get_thread_num(), nthreads = omp_get_num_threads();

    printf("Greetings from CPU thread %d/%d of process %d/%d!\n",
	   tid, nthreads, rank, nprocs);
  }
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) printf("Time: %f\n", MPI_Wtime() - start);
  MPI_Finalize();
}
