#include <mpi.h>
#include <stdio.h>
#define comm MPI_COMM_WORLD

int myrank, nprocs;

/* each non-root process sends a message to root; root receives using
 * MPI_ANY_SOURCE.  This is a perfectly fine deadlock-free function. */
void f() {
  if (myrank == 0) {
    MPI_Status status;
    int x;
    
    for (int i = 1; i < nprocs; i++) {
      MPI_Recv(&x, 1, MPI_INT, MPI_ANY_SOURCE, 0, comm, &status);
      printf("Proc 0: received %d from proc %d\n", x, status.MPI_SOURCE);
      fflush(stdout);
    }
  } else {
    MPI_Send(&myrank, 1, MPI_INT, 0, 0, comm);
  }
}

/* each non-root process sends a message to root; root receives in
 * order of increasing rank.  This is a perfectly fine deadlock-free
 * function. */
void g() {
  if (myrank == 0) {
    MPI_Status status;
    int x;
    
    for (int i = 1; i < nprocs; i++) {
      MPI_Recv(&x, 1, MPI_INT, i, 0, comm, &status);
      printf("Proc 0: received %d from proc %d\n", x, status.MPI_SOURCE);
      fflush(stdout);
    }
  } else {
    MPI_Send(&myrank, 1, MPI_INT, 0, 0, comm);
  }
}

/* what happens when I call the two deadlock-free functions in
   sequence? */
int main() {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(comm, &nprocs);
  MPI_Comm_rank(comm, &myrank);
  f();
  MPI_Barrier(comm);
  g();
  MPI_Finalize();
}
