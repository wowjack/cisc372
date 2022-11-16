#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#ifndef N
#define N 20
#endif
#define TYPE MPI_UNSIGNED

int nprocs; // number of processes
int myrank; // my rank in MPI_COMM_WORLD
int first; // global index of first cell I own
int n_local; // how many cells do I "own"?
/* Pointers to current and previous arrays.   These have length
 * n_local+2: an extra "ghost" cell on the left and right to mirror
 * values on the left and right neighbors */
unsigned int *p, *q;
int left, right; // my left and right neighbors (ranks)

// n=number of elements, p=number of procs, r=rank
#define FIRST(r) ((2*N+1)*(r)/nprocs)
#define NUM_OWNED(r) (FIRST(r+1) - FIRST(r))
#define OWNER(j) ((nprocs*((j)+1)-1)/(2*N+1))
#define LOCAL_INDEX(j) ((j)-FIRST(OWNER(j)))

void print_block(unsigned int *buf, int length) {
  for (int j=0; j<length; j++) {
    if (buf[j]==0)
      printf("      ");
    else 
      printf("%6u", buf[j]);
  }
}

/* Proc 0 receives a block from each proc and prints it, in order.
 * Proc 0 uses q as a temporary buffer. */
void print() {
  if (myrank == 0) {
    MPI_Status status;
    int count;
    
    print_block(p+1, n_local);
    for (int i=1; i<nprocs; i++) {
      MPI_Recv(q, n_local+1, TYPE, i, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, TYPE, &count);
      print_block(q, count);
    }
    printf("\n");
    q[0] = q[n_local+1] = 0;
  } else {
    MPI_Send(p+1, n_local, TYPE, 0, 0, MPI_COMM_WORLD);
  }
}

/* When this is called p holds the current values, and when it returns,
 * p will hold the next set of values. */
void update() {
  for (int j=1; j<=n_local; j++)
    q[j] = p[j-1]+p[j+1];
  unsigned int *tmp = p; p=q; q=tmp; // swap p and q
}

/* Exchange "ghost cells".
 * Send p[1] to left and p[n_local] to right.
 * Recv into p[n_local+1] from right and into p[0] from left. */
void exchange_ghosts() {
  MPI_Sendrecv(p+1, 1, TYPE, left, 0,
	       p+(n_local+1), 1, TYPE, right, 0,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(p+n_local, 1, TYPE, right, 0,
	       p, 1, TYPE, left, 0,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

int main() {
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  first = FIRST(myrank);
  n_local = NUM_OWNED(myrank);
  p = malloc((n_local+2)*sizeof(unsigned int));
  q = malloc((n_local+2)*sizeof(unsigned int));
  for (int i=0; i<n_local+2; i++) p[i] = q[i] = 0;
  if (OWNER(N) == myrank) p[1+LOCAL_INDEX(N)] = 1;
  left = n_local == 0 || first == 0 ? MPI_PROC_NULL : OWNER(first-1);
  right = n_local == 0 || first+n_local >= 2*N+1 ? MPI_PROC_NULL :
    OWNER(first+n_local);
#ifdef DEBUG
  if (myrank == 0) {
    printf("N=%d, OWNER(N)=%d, LOCAL_INDEX(N)=%d\n",
	   N, OWNER(N), LOCAL_INDEX(N));
    fflush(stdout);
  }
  for (int i=0; i<nprocs; i++) {
    if (i==myrank) {
      printf("rank=%d: left=%d, right=%d, n_local=%d, first=%d\n",
	     myrank, left, right, n_local, first);
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  sleep(1); // wait one second
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  for (int i=0; i<N; i++) {
    print();
    exchange_ghosts();
    update();
  }
  print();
  MPI_Finalize();
  free(p);
  free(q);
}
