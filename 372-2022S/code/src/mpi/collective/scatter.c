/* Example demonstrating MPI_Scatter.  Proc 0 has an array of length
   2*nprocs.   The contents of this array are scattered among the procs,
   so each proc receives 2 items.
   
   int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
       void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main() {
  int rank, nprocs, nl = 2;
  float * data, data_l[nl];

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    data = (float*) malloc(nprocs * sizeof(float) * nl);
    for (int i = 0; i < nprocs * nl; i++)
      data[i] = i;
  }
  MPI_Scatter(data, nl, MPI_FLOAT, data_l, nl, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if (rank == 0) free(data);
  for (int n = 0; n < nprocs; n++) {
    if (rank == n) {
      printf("Proc %d: ", rank);
      for (int j = 0; j < nl; j++)
	printf("%f ", data_l[j]);
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();
}
