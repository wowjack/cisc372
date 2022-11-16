/* "Hello, world" using MPI+Pthreads.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <pthread.h>

const int nthreads = 2; // number of threads per proc
int nprocs;
int rank;

/* The thread function.
 * The signature must be void* -> void*
 */
void* hello(void* arg) {
  // always ok to cast from T* to void* and back, for any object type T...
  int * tidp = (int*)arg; 

  printf("Hello from Pthread thread %d/%d of process %d/%d!\n",
	 *tidp, nthreads, rank, nprocs);
  return NULL;
}

int main () {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  pthread_t threads[nthreads];
  int tids[nthreads];
  
  for (int i=0; i<nthreads; i++)
    tids[i] = i;
  for (int i=0; i<nthreads; i++)
    pthread_create(threads + i, NULL, hello, tids + i);
  for (int i=0; i<nthreads; i++)
    pthread_join(threads[i], NULL);
  MPI_Finalize();
}
