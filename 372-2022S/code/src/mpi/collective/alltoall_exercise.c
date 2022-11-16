/* Exercise: alltoall.  Each proc has an array of ints of length
   nprocs.  On proc i, the elements of this array are all 2*i.
   Transpose this distributed array and print the result on the last
   proc.
   */
#include <stdio.h>
#include <mpi.h>

int main() {
  int rank, nprocs;
  
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int sendBuffer[nprocs], recvBuffer[nprocs];
  
  for (int i = 0; i < nprocs; i++)
    sendBuffer[i] = 2*rank;
  MPI_Alltoall(sendBuffer, 1, MPI_INT, recvBuffer, 1, MPI_INT, MPI_COMM_WORLD);
  if (rank == nprocs-1) {
    printf("Proc %d received: ", rank);
    for (int i = 0; i < nprocs; i++)
      printf("%d ", recvBuffer[i]);
    printf("\n");
    fflush(stdout);
  }
  MPI_Finalize();
}
