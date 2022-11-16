/* Implementation of a "flag barrier" using pthreads.  This is a
   reuseable n-thread barrier.  A single flag is used for each thread.
   A shared variable count is used to keep track of the number of
   threads currently in the barrier.  When a thread enters it
   increments counts and waits for a signal on its flag.  The last
   thread to enter signals everyone to leave and sets count back to 0.
 
   Author  : Stephen F. Siegel
   Date    : 2016-oct-24
   Revised : 2020-oct-14
 */
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "flag.h"
#include "barrier.h"

struct barrier_state {
  /* Number of threads in team using this barrier */
  int nthreads;
  /* guards access to count, in */
  pthread_mutex_t lock;
  /* how many threads are currently in this barrier? */
  int count;
  /* signals to leave the barrier */
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
  bs->count = 0;
  bs->depart = malloc(nthreads*sizeof(flag_t));
  assert(bs->depart);
  for (int i=0; i<nthreads; i++)
    flag_init(&bs->depart[i], 0);
  pthread_mutex_init(&bs->lock, NULL);
  return bs;
}

void barrier_state_destroy(barrier_state_t bs) {
  for (int i=0; i<bs->nthreads; i++)
    flag_destroy(&bs->depart[i]);
  free((void*)bs->depart);
  pthread_mutex_destroy(&bs->lock);
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
  
  pthread_mutex_lock(&bs->lock);
  if (bs->count == nthreads-1) { // I am last to enter barrier
    for (int i=0; i<nthreads; i++) { // release everyone else
      if (i!=tid) flag_raise(&bs->depart[i]);
    }
    bs->count = 0;
    pthread_mutex_unlock(&bs->lock);
  } else { // enter barrier and wait for signal to leave...
    bs->count++;
    pthread_mutex_unlock(&bs->lock);
    flag_lower(&bs->depart[tid]);
  }
}
