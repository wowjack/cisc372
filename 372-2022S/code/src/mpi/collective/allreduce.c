/* Example demonstrating MPI_Allreduce.  Every proc has an array of
   length 5.  They are reduced with the result going to all.

   int MPI_Allreduce(void *sendbuf, void *recvbuf, int count,
       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
 */
#include <stdio.h>
#include <mpi.h>

int rank, nprocs, n = 5;

void print_bufs(float * buf) {
  for (int p = 0; p < nprocs; p++) {
    if (rank == p) {
      printf("Proc %d has: ", p);
      for (int i = 0; i < n; i++)
	printf("%8.1f ", buf[i]);
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  } 
}

int main() {
  float sbuf[n], rbuf[n];

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int i = 0; i < n; i++)
    sbuf[i] = 100*rank+i;
  if (rank == 0)
    printf("Send buffers:\n");
  MPI_Barrier(MPI_COMM_WORLD);
  print_bufs(sbuf);
  MPI_Allreduce(sbuf, rbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    printf("Recv buffers:\n");
  MPI_Barrier(MPI_COMM_WORLD);
  print_bufs(rbuf);
  MPI_Finalize();
}
