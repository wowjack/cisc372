/* Hello world in MPI.  
 * Compile: mpicc -o hello hello.c
 * Execute: mpiexec -n 8 ./hello (to run with 8 processes)
 */
#include<stdio.h>
#include<mpi.h>

int main() {
  int rank; // my rank
  int nprocs; // number of procs

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  for (int i=0; i<nprocs; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == i) {
      printf("Hello from MPI process %d of %d!\n", rank, nprocs);
      fflush(stdout);
    }
  }
  MPI_Finalize();
}
