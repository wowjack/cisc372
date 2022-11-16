/* Demo of MPI_Alltoall.  Each proc has an array of length nprocs
 * in which every entry is the proc's rank.  They do an all-to-all.
 *
 * int MPI_Alltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype,
 *   void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) 
 */
#include <stdio.h>
#include <mpi.h>

int main() {
  int rank, nprocs;
  
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  float sendBuffer[nprocs], recvBuffer[nprocs];
  
  for (int i = 0; i < nprocs; i++)
    sendBuffer[i] = rank;
  for (int n = 0; n < nprocs; n++) {
    if (rank == n) {
      printf("Proc %d sending: ", rank);
      for (int j = 0; j < nprocs; j++)
	printf("%f ", sendBuffer[j]);
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Alltoall(sendBuffer, 1, MPI_FLOAT, recvBuffer, 1, MPI_FLOAT,
	       MPI_COMM_WORLD);
  for (int n = 0; n < nprocs; n++) {
    if (rank == n) {
      printf("Proc %d received: ", rank);
      for (int j = 0; j < nprocs; j++)
	printf("%f ", recvBuffer[j]);
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();
}
