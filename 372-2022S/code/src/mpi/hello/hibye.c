#include <stdio.h>
#include <mpi.h>

int main() {
  int rank;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("Hello from process %d.\n", rank);
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  printf("Goodbye from process %d.\n", rank);
  fflush(stdout);
  MPI_Finalize();
}
