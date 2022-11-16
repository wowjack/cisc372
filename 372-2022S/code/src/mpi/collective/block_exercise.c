/* Exercise: Block distribution of array, gathered onto proc 0.  One
   command line argument: the length n of the array. */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(j,p,n) (((p)*((j)+1)-1)/(n))
#define CEILING(i,j)       (((i)+(j)-1)/(j))

int main(int argc, char * argv[]) {
  int rank, nprocs, * recvcounts, * displs, nl, n, first;
  int * data = NULL, * data_l;
 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(argc == 2);
  n = atoi(argv[1]);
  if (rank == 0) {
    printf("n = %d, nprocs = %d\n", n, nprocs);
    fflush(stdout);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  first = BLOCK_LOW(rank, nprocs, n);
  nl = BLOCK_SIZE(rank, nprocs, n);
  data_l = (int*) malloc(nl * sizeof(int));
  for (int i = 0; i < nl; i++)
    data_l[i] = i + first;
  for (int p = 0; p < nprocs; p++) {
    if (rank == p) {
      printf("Proc %d: first = %d, nl = %d, data_l = { ",
	     p, first, nl);
      for (int i = 0; i < nl; i++)
	printf("%d ", data_l[i]);
      printf("}\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (rank == 0) {
    data = (int*) malloc(n * sizeof(int));
    recvcounts = (int*) malloc(nprocs * sizeof(int));
    displs = (int*) malloc(nprocs * sizeof(int));
    for (int i = 0; i < nprocs; i++)
      recvcounts[i] = BLOCK_SIZE(i, nprocs, n);
    displs[0] = 0;
    for (int i = 1; i < nprocs; i++)
      displs[i] = displs[i-1] + recvcounts[i-1];
  }
  MPI_Gatherv(data_l, nl, MPI_INT, data, recvcounts, displs,
	      MPI_INT, 0, MPI_COMM_WORLD);
  free(data_l);
  if (rank == 0) {
    printf("Result on proc 0: { ");
    for (int i = 0; i < n; i++)
      printf("%d ", data[i]);
    printf("}\n");
    free(displs);
    free(recvcounts);
    free(data);
  }
  MPI_Finalize();
}
