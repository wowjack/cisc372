#include<stdio.h>
#include<mpi.h>

int main() {
  int nprocs, rank;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  const int right = rank < nprocs - 1 ? rank + 1 : MPI_PROC_NULL,
    left = rank > 0 ? rank - 1 : MPI_PROC_NULL;
  int rbuf, sbuf = 100 + rank;
  MPI_Sendrecv(&sbuf, 1, MPI_INT, right, 0, &rbuf, 1, MPI_INT, left, 0,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (rank > 0) printf("Proc %d: received %d\n", rank, rbuf);
  MPI_Finalize();
}
