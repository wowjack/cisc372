/* Demonstration of MPI_Bcast.  Proc 0 has a buffer of length 10,
   which is broadcasts to all procs.

   int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
                 MPI_Comm comm)
 */
#include <stdio.h>
#include <mpi.h>

int main() {
  int rank, nprocs, n = 10;
  float buf[n];
 
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    for (int i = 0; i < n; i++)
      buf[i] = i;
  MPI_Bcast(buf, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  for (int p = 0; p < nprocs; p++) {
    if (rank == p) {
      printf("Proc %d has: ", rank);
      for (int i = 0; i < n; i++)
	printf(" %f", buf[i]);
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();
}
