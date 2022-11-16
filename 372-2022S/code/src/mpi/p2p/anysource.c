#include<stdio.h>
#include<mpi.h>

int main() {
  int message, rank;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    message = 0; MPI_Send(&message, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
  } else if (rank == 1)  {
    message = 1; MPI_Send(&message, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
  } else if (rank == 2) {
    for (int i=0; i<2; i++) {
      MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("Proc 2 received: %d\n", message);
    }
  }
  MPI_Finalize();
}
