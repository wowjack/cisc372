#include <mpi.h>
#include <stdio.h>

int main() {
  int rank, myNumber, otherNumber;
  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    myNumber = 10;
    MPI_Sendrecv(&myNumber, 1, MPI_INT, 1, 99,
		 &otherNumber, 1, MPI_INT, 1, 99,
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else if (rank==1) {
    myNumber = 20;
    MPI_Sendrecv(&myNumber, 1, MPI_INT, 0, 99,
		 &otherNumber, 1, MPI_INT, 0, 99,
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  if (rank < 2)
    printf("Process %d: received %d\n", rank, otherNumber);
  MPI_Finalize();
}
