/* Implementation of a "counter barrier" using pthreads.  Uses a
   condition variable to broadcast barrier release.  This is a
   reuseable n-thread barrier.  Based on Peter Pacheco, An
   Introduction to Parallel Programming, Elsevier, 2011, Sec. 4.8.3
   "Condition Variables".
 
   Author    : Stephen F. Siegel
   Date      : 2020-oct-14
   Modified  : 2020-oct-14
 */
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "barrier.h"

struct barrier_state {
  /* The number of threads */
  int nthreads;
  /* Guards access to barrier_count, barrier_sig */
  pthread_mutex_t lock;
  /* how many are in barrier? */
  int count;
  /* used to signal barrier release */
  pthread_cond_t sig;
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
  pthread_mutex_init(&bs->lock, NULL);
  pthread_cond_init(&bs->sig, NULL);
  return bs;
}

void barrier_state_destroy(barrier_state_t bs) {
  pthread_mutex_destroy(&bs->lock);
  pthread_cond_destroy(&bs->sig);
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
  
  pthread_mutex_lock(&bs->lock);
  bs->count++;
  if (bs->count == bs->nthreads) { // I am last to enter barrier
    bs->count = 0;
    pthread_cond_broadcast(&bs->sig);
  } else { // wait for signal to leave...
    while (pthread_cond_wait(&bs->sig, &bs->lock) != 0);
  }
  pthread_mutex_unlock(&bs->lock);
}
