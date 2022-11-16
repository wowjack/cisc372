/* Demonstration of MPI_Gatherv.  Proc of rank i has buffer of length i
 * consisting of just i's.  These are gathered onto the root (proc 0).
 * Syntax for MPI_Gatherv:
 *  MPI_Gatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype,
 *    void* recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype,
 *    int root, MPI_Comm comm) 
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(){
  int rank, nprocs, * recvcounts, * displs, total_size, nl;
  float * data = NULL, * data_l;
 
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  nl = rank;
  data_l = (float*) malloc(nl * sizeof(float));
  for (int i = 0; i < nl; i++)
    data_l[i] = rank;
  if (rank == 0)  {
    total_size = (nprocs*(nprocs-1))/2;
    data = (float*) malloc(total_size * sizeof(float));
    recvcounts = (int*) malloc(nprocs * sizeof(int));
    displs = (int*) malloc(nprocs * sizeof(int));
    for (int i = 0; i < nprocs; i++)
      recvcounts[i] = i;
    displs[0] = 0;
    for (int i = 1; i < nprocs; i++)
      displs[i] = displs[i-1] + recvcounts[i-1];
  }
  MPI_Gatherv(data_l, nl, MPI_FLOAT, data, recvcounts, displs,
	      MPI_FLOAT, 0, MPI_COMM_WORLD);
  free(data_l);
  if (rank == 0) {
    for (int i = 0; i < total_size; i++)
      printf("%f ", data[i]);
    printf("\n");
    free(displs);
    free(recvcounts);
    free(data);
  }
  MPI_Finalize();
}
