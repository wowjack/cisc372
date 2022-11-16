
#include <mpi.h>
#include <stdio.h>

int myrank, nprocs;

void f() {
  int x;

  if (myrank == 0) {
    MPI_Status status;

    for (int i = 1; i < nprocs; i++) {
      MPI_Recv(&x, 1, MPI_INT, MPI_ANY_SOURCE,
	       0, MPI_COMM_WORLD, &status);
      printf("Proc 0: received %d from proc %d\n", x, status.MPI_SOURCE);
      MPI_Send(&x, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
      fflush(stdout);
    }
  } else {
    x = myrank;
    MPI_Send(&x, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(&x, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Proc %d: received %d from proc 0\n", myrank, x);
    fflush(stdout);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
 
  f();
  f();

  MPI_Finalize();
}
