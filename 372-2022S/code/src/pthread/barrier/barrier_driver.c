/* Barrier driver.  Tests a barrier with assertions.  Take 2 command
   line args: nthreads, niter. */
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "barrier.h"

barrier_state_t bs;
int niter; // number of times to loop
int nthreads; // number of threads (from command line arg)
int * counters; // for testing

void * driver(void * arg) {
  int tid = *((int*)arg);
  barrier_t bar = barrier_create(bs, tid);
  
  for (int i=0; i<niter; i++) {
    counters[tid] = i;
#ifdef DEBUG
    printf("Thread %d entering barrier %d.\n", tid, i); fflush(stdout);
#endif
    barrier_wait(bar);
    for (int j=0; j<nthreads; j++)
      assert(counters[j]==i);
#ifdef DEBUG
    printf("Thread %d leaving barrier %d.\n", tid, i); fflush(stdout);
#endif
    barrier_wait(bar);
  }
#ifdef DEBUG
  printf("Thread %d finished all barriers.\n", tid); fflush(stdout);
#endif
  barrier_destroy(bar);
  return NULL;
}

/* Arg1: nthreads.  Arg2: niter. */
int main(int argc, char * argv[]) {
  assert(argc == 3);
  nthreads = atoi(argv[1]);
  assert(nthreads >= 1);
  niter = atoi(argv[2]);
  assert(niter >= 0);
  counters = malloc(nthreads*sizeof(int));
  assert(counters);
  bs = barrier_state_create(nthreads);

  int tids[nthreads];
  pthread_t threads[nthreads];

  for (int i=0; i<nthreads; i++)
    tids[i] = i;
  for (int i=0; i<nthreads; i++)
    pthread_create(threads + i, NULL, driver, tids + i);
  for (int i=0; i<nthreads; i++) {
    printf("Main thread: waiting for thread %d...\n", i); fflush(stdout);
    pthread_join(threads[i], NULL);
    printf("Main thread: joined thread %d.\n", i); fflush(stdout);
  }
  barrier_state_destroy(bs);
  free(counters);
}
