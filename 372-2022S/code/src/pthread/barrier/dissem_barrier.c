/* Dissemination barrier.  A symmetric barrier. Based on Hensgen,
   Finkel, and Manber, "Two Algorithms for Barrier Synchronization",
   International Journal of Parallel Programming, Vol. 17, No. 1, 1988.

   This is a symmetric barrier.  Two flags are used for each thread,
   at each stage.  A full 2-thread barrier is used for each thread
   pair.  This is pretty much a direct implementation of the
   pseudocode given in the paper.

   Author   : Stephen F. Siegel
   Date     : 2020-oct-14
   Modified : 2020-nov-23
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "barrier.h"
#include "flag.h"

struct barrier_state {
  int nthreads;
  /* ceil(log_2(nthreads)) */
  int nstages;
  /* Array of length nstages of array of length nthreads of flag */
  flag_t ** a;
  /* Second array like above. */
  flag_t ** b;
};

struct barrier {
  barrier_state_t bs;
  int tid;
};

/* Allocates 2d array of flags: flags[n][m], initializing all n*m
   flags to 0. */
flag_t ** allocate_flags(int n, int m) {
  if (n == 0) return NULL;
  
  flag_t ** result = malloc(n*sizeof(flag_t*));
  flag_t * storage = malloc(n*m*sizeof(flag_t));

  assert(result);
  assert(storage);
  for (int i=0; i<n; i++)
    result[i] = &storage[i*m];
  for (int i=0; i<n*m; i++)
    flag_init(&storage[i], 0);
  return result;
}

/* Frees 2d-array of flags */
void free_flags(flag_t ** flags) {
  if (flags == NULL) return;
  free((void*)flags[0]);
  free(flags);
}

barrier_state_t barrier_state_create(int nthreads) {
  barrier_state_t bs = malloc(sizeof(struct barrier_state));
  
  assert(nthreads >= 1);
  assert(bs);
  bs->nthreads = nthreads;
  bs->nstages = 0;
  for (int i=1; i<nthreads; i*=2)
    bs->nstages++;
  bs->a = allocate_flags(bs->nstages, nthreads);
  bs->b = allocate_flags(bs->nstages, nthreads);
  return bs;
}

void barrier_state_destroy(barrier_state_t bs) {
  free_flags(bs->a);
  free_flags(bs->b);
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
  const int tid = bar->tid, nthreads = bs->nthreads, nstages = bs->nstages;
  
  for (int stage=0, i=1; stage<nstages; stage++, i*=2) {
    flag_raise(&bs->a[stage][(tid+i)%nthreads]);
    flag_lower(&bs->a[stage][tid]);
    flag_raise(&bs->b[stage][tid]);
    flag_lower(&bs->b[stage][(tid+i)%nthreads]);
  }
}
