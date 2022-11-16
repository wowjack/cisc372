#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#ifndef N
#define N 20
#endif
// Standard block distribution scheme: N items distributed over nprocs procs
#define FIRST(r) ((N)*(r)/nprocs)
#define NUM_OWNED(r) (FIRST((r)+1) - FIRST(r))
#define OWNER(j) ((nprocs*((j)+1)-1)/(N))
#define LOCAL_INDEX(j) ((j)-FIRST(OWNER(j)))

int main() {
  int nprocs, rank; // number of procs, rank of this proc
  int first;          // global index of first cell owned by this proc
  int n_local;        // number of cells owned by this proc
  
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  first = FIRST(rank);
  n_local = NUM_OWNED(rank);
#ifdef DEBUG
  printf("Rank %d: first=%d, n_local=%d\n", rank, first, n_local);
#endif

  unsigned int a[n_local]; // local block of global array a
  unsigned long sum = 0, global_sum;

  for (int i=0; i<n_local; i++) {
    const int j = first + i; // convert from local to global index
    a[i] = j * j;
  }
  for (int i=0; i<n_local; i++)
    sum += a[i];
  MPI_Reduce(&sum, &global_sum, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  if (rank == 0) printf("sum = %ld\n", global_sum);
}
