#include<stdio.h>
#include<mpi.h>

int main() {
  int message, rank;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    message = 173;
    MPI_Send(&message, 1, MPI_INT, 1, 9, MPI_COMM_WORLD);
  } else if (rank == 1)  {
    printf("Proc 1: was I supposed to do something?\n");
  }
  MPI_Finalize();
}
