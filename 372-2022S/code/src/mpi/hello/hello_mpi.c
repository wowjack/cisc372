/* Hello world in MPI.  
 * Compile: mpicc -std=c11 -o hello hello.c
 * Execute: srun -n 8 ./hello (to run with 8 processes on cisc372.cis.udel.edu)
 * or: mpiexec -n 8 ./hello (on your own machine)
 */
#include<stdio.h>
#include<mpi.h>

int main() {
  int rank, nprocs;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  printf("Hello from MPI process %d of %d!\n", rank, nprocs);
  fflush(stdout);
  MPI_Finalize();
}
