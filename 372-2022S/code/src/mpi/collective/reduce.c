/* Example demonstrating MPI_Reduce.  Every proc has an array of length 5.
   They are reduced with the result going to proc 0.

   int MPI_Reduce(void *sendbuf, void *recvbuf, int count,
        MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)  
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main() {
  int rank, nprocs, n = 5;
  float sbuf[n], *rbuf;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  rbuf = rank == 0 ? malloc(n*sizeof(float)) : NULL;
  for (int i=0; i<n; i++)
    sbuf[i] = 100*rank+i;
  for (int p=0; p<nprocs; p++) {
    if (rank == p) {
      printf("Proc %d sbuf: ", p);
      for (int i=0; i<n; i++)
	printf("%8.1f ", sbuf[i]);
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Reduce(sbuf, rbuf, n, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Proc 0 rbuf: ");
    for (int i=0; i<n; i++) printf("%8.1f ", rbuf[i]);
    printf("\n");
  }
  MPI_Finalize();
  if (rank == 0)
    free(rbuf);
}
