/* A coordinator barrier.  Thread 0 is the coordinator.  There are two
   flags for each thread.  When a thread arrives, it raises its arrive
   flag and waits on its depart flag.  The coordinator waits on and
   lowers the arrive flag of each thread, then raises all the depart
   flags. */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "flag.h"
#include "barrier.h"

struct barrier_state {
  /* Number of threads in team using this barrier */
  int nthreads;
  /* signals arrival at the barrier.  length nthreads */
  flag_t * arrive;
  /* signals to leave the barrier.  length nthreads */
  flag_t * depart;
};

struct barrier {
  barrier_state_t bs;
  int tid;
};

barrier_state_t barrier_state_create(int nthreads) {
  barrier_state_t bs = malloc(sizeof(struct barrier_state));
  
  assert(bs);
  bs->nthreads = nthreads;
  bs->arrive = malloc(nthreads*sizeof(flag_t));
  assert(bs->arrive);
  bs->depart = malloc(nthreads*sizeof(flag_t));
  assert(bs->depart);
  for (int i=0; i<nthreads; i++) {
    flag_init(&bs->arrive[i], 0);
    flag_init(&bs->depart[i], 0);
  }
  return bs;
}

void barrier_state_destroy(barrier_state_t bs) {
  for (int i=0; i<bs->nthreads; i++) {
    flag_destroy(&bs->arrive[i]);
    flag_destroy(&bs->depart[i]);
  }
  free((void*)bs->arrive);
  free((void*)bs->depart);
  free(bs);
}

barrier_t barrier_create(barrier_state_t bs, int tid) {
  assert (0<=tid && tid<bs->nthreads);
  barrier_t bar = malloc(sizeof(struct barrier));
  assert(bar);
  bar->tid = tid;
  bar->bs = bs;
  return bar;
}

void barrier_destroy(barrier_t bar) {
  free(bar);
}

void barrier_wait(barrier_t bar) {
  const barrier_state_t bs = bar->bs;
  const int tid = bar->tid;
  const int nthreads = bs->nthreads;

  if (tid == 0) { // I am the coordinator
    for (int i=1; i<nthreads; i++)
      flag_lower(&bs->arrive[i]);
    for (int i=1; i<nthreads; i++)
      flag_raise(&bs->depart[i]);
  } else  {
    flag_raise(&bs->arrive[tid]);
    flag_lower(&bs->depart[tid]);
  }
}
