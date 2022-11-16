/* tags.c: demonstration of receiving messages out of order using
   tags.  Note that this program is not safe --- technically, it could
   deadlock.  But if it does not deadlock, the messages will be
   received in the reverse order.  */
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
    MPI_Recv(&message, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Proc 1 received: %d\n", message); 
    MPI_Recv(&message, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Proc 1 received: %d\n", message); 
  }
  MPI_Finalize();
}
