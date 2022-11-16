/* Demonstration of MPI_Allgatherv.  Proc of rank i has buffer of
   length i consisting of just i's.  These are gathered onto every
   proc.  Syntax for MPI_Allgatherv:
 
   int MPI_Allgatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype,
     void* recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype,
     MPI_Comm comm)
*/
#include <stdio.h>
#include <mpi.h>

int main(){
  int rank, nprocs, total_size, nl;
 
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  nl = rank;
  total_size = (nprocs*(nprocs-1))/2;

  int recvcounts[nprocs], displs[nprocs];
  float data_l[nl], data[total_size];

  for (int i = 0; i < nl; i++)
    data_l[i] = rank;
  for (int i = 0; i < nprocs; i++)
    recvcounts[i] = i;
  displs[0] = 0;
  for (int i = 1; i < nprocs; i++)
    displs[i] = displs[i-1] + recvcounts[i-1];
  MPI_Allgatherv(data_l, nl, MPI_FLOAT, data, recvcounts, displs,
		 MPI_FLOAT, MPI_COMM_WORLD);
  for (int n = 0; n < nprocs; n++) {
    if (rank == n) {
      printf("Proc %d: ", rank);
      for (int i = 0; i < total_size; i++)
	printf("%f ", data[i]);
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();
}
