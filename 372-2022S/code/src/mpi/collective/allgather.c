/* Example demonstrating MPI_Allgather.  All procs have buffer of
   length 2.  These are gathered together on every proc.  Syntax:
 
   int MPI_Allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
        void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) 
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main() {
  int rank, nprocs, nl = 2;
  
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  float data[nprocs*nl], data_l[nl];

  for (int i = 0; i < nl; i++)
    data_l[i] = rank;
  MPI_Allgather(data_l, nl, MPI_FLOAT, data, nl, MPI_FLOAT, MPI_COMM_WORLD);
  for (int n = 0; n < nprocs; n++) {
    if (rank == n) {
      printf("Proc %d: ", rank);
      for (int j = 0; j < nl*nprocs; j++)
	printf("%f ", data[j]);
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();
}
