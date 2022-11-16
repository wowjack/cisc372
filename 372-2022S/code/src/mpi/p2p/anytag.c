/* anytag: the messages will be received in the order sent.  The MPI_ANY_TAG recv
   must match the oldest message sent from proc 0 */
#include<stdio.h>
#include<mpi.h>

int main() {
  int message, rank;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    message = 1;
    MPI_Send(&message, 1, MPI_INT, 1, 1, MPI_COMM_WORLD); // tag=1
    message = 2;
    MPI_Send(&message, 1, MPI_INT, 1, 2, MPI_COMM_WORLD); // tag=2
  } else if (rank == 1)  {
    MPI_Recv(&message, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Proc 1 received: %d\n", message); 
    MPI_Recv(&message, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Proc 1 received: %d\n", message); 
  }
  MPI_Finalize();
}
