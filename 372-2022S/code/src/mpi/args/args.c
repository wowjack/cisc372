#include <stdio.h>
#include <mpi.h>

int main(int argc, char * argv[]) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("Proc %d: argc=%d, ", rank, argc);
  for (int i=1; i<argc; i++) printf("%s ", argv[i]);
  printf("\n");
  MPI_Finalize();
}
