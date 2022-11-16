/* MPI/OpenMP hybrid hello world.  One command line argument: number
   of threads per process. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char * argv[]) {
  int nprocs; // number of MPI processes
  int rank; // rank of this process
  int nthreads_requested; // number of threads per process requested
  int thread_support; // level of thread support provided

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support);
  assert(argc==2);
  nthreads_requested = atoi(argv[1]);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("Proc %d: Thread support level requested: %d, provided: %d\n",
	 rank, MPI_THREAD_FUNNELED, thread_support);
  fflush(stdout);
#pragma omp parallel num_threads(nthreads_requested)
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    printf("Hello from thread %d/%d of process %d/%d!\n",
           tid, nthreads, rank, nprocs);
    fflush(stdout);
  }
  MPI_Finalize();
}
