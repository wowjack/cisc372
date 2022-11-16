// assumes nprocs | N.  Each procs owns N/nprocs cells.
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <assert.h>

#ifndef N
#define N 20
#endif

int nprocs;
int myrank;
int first;
int n_local;

#define FIRST(r) ((r)*(N/nprocs))
#define NUM_OWNED(r) (N/nprocs)
#define OWNER(j) ((j)/(N/nprocs))
#define LOCAL_INDEX(j) ((j)%(N/nprocs))

unsigned int *a;

int main() {
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  assert(N%nprocs==0);
  first = FIRST(myrank);
  n_local = NUM_OWNED(myrank);
  a = malloc(n_local*sizeof(unsigned int));
#ifdef DEBUG
  printf("Rank %d: first=%d, n_local=%d\n", myrank, first, n_local);
#endif

  unsigned long sum = 0, global_sum;

  for (int i=0; i<n_local; i++) {
    int j = first+i; // convert from local to global index
    a[i] = j * j;
  }
  for (int i=0; i<n_local; i++)
    sum += a[i];
  MPI_Reduce(&sum, &global_sum, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  if (myrank == 0) {
    printf("sum = %ld\n", global_sum);
  }
  free(a);
}
