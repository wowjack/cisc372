/* simplend.c : simple example of nondeterminism resulting from
 * use of MPI_ANY_SOURCE.  Run me with 3 procs.  Sometimes you will
 * see one thing, sometimes another.
 */
#include <mpi.h>
#include <stdio.h>

int nprocs;
int myrank;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (myrank == 0) {
    MPI_Status status;

    MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, 99,
	     MPI_COMM_WORLD, &status);
    MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, 99,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("%d\n", status.MPI_SOURCE);
    fflush(stdout);
  } else if (myrank <= 2) {
    MPI_Send(NULL, 0, MPI_INT, 0, 99, MPI_COMM_WORLD);
  }
  MPI_Finalize();
}
