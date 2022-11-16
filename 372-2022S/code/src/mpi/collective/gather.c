/* Demonstration of MPI_Gather.  All procs have buffer of length 2.
   These buffers are gathered onto proc 0.  Syntax:

   int MPI_Gather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
     void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
     MPI_Comm comm) 
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main() {
  int rank, nprocs, nl = 2;
  float data_l[nl], * data = NULL;
 
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int i = 0; i < nl; i++)
    data_l[i] = rank;
  if (rank == 0)
    data = (float*) malloc(nprocs * sizeof(float) * nl);
  MPI_Gather(data_l, nl, MPI_FLOAT, data, nl, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (int i = 0; i < nl*nprocs; i++)
      printf("%f ", data[i]);
    printf("\n");
    free(data);
  }
  MPI_Finalize();
}
