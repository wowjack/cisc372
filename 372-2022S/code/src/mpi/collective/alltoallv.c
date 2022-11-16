/* Demo of MPI_Alltoallv.
 *
 * Proc i sends j things to proc j, for a total of n(n-1)/2
 *    the things it sends are just its rank.
 * Proc j receives j things from each proc, for a total of jn things.
 *
 * int MPI_Alltoallv(void* sendbuf, int *sendcounts, int *sdispls,
 *   MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *rdispls,
 *   MPI_Datatype recvtype, MPI_Comm comm) 
 */

#include <stdio.h>
#include <mpi.h>

int main() {
  int rank, nprocs, sendBufLen, recvBufLen;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  sendBufLen = (nprocs*(nprocs-1))/2;
  recvBufLen = rank*nprocs;

  int sendcounts[nprocs], sdispls[nprocs], recvcounts[nprocs], rdispls[nprocs];
  int sendBuffer[sendBufLen], recvBuffer[recvBufLen];
  
  for (int i = 0; i < nprocs; i++)
    sendcounts[i] = i;
  sdispls[0] = 0;
  for (int i = 1; i < nprocs; i++)
    sdispls[i] = sdispls[i-1] + sendcounts[i-1];
  for (int i = 0; i < nprocs; i++)
    recvcounts[i] = rank;
  rdispls[0] = 0;
  for (int i = 1; i < nprocs; i++)
    rdispls[i] = rdispls[i-1] + recvcounts[i-1];
  for (int i = 0; i < sendBufLen; i++)
    sendBuffer[i] = rank;
  for (int n = 0; n < nprocs; n++) {
    if (rank == n) {
      printf("Proc %d sending: ", rank);
      for (int j = 0; j < sendBufLen; j++)
	printf("%d ", sendBuffer[j]);
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Alltoallv(sendBuffer, sendcounts, sdispls, MPI_INT,
		recvBuffer, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
  for (int n = 0; n < nprocs; n++) {
    if (rank == n) {
      printf("Proc %d received: ", rank);
      for (int j = 0; j < recvBufLen; j++)
	printf("%d ", recvBuffer[j]);
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();
}
