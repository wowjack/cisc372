/* Simple solution to critical section problem using semaphores */

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "sem.h"

int niter = 10;
int nthreads;
semaphore s;


void init_critical() {
  semaphore_init(&s, 1);
}


void finalize_critical() {
  semaphore_destroy(&s);
}


void enter_critical(int tid) {
  // P(s) : waitsfor s>0 then decrements s
  semaphore_P(&s);
}

void exit_critical(int tid) {
  // V(s) : increments s
  semaphore_V(&s);
}

void * thread(void * arg) {
  int tid = *((int*)arg);
  
  for (int i=0; i< niter; i++) {
    enter_critical(tid);
    printf("Thread %d is in the critical section\n", tid);
    fflush(stdout);
    exit_critical(tid);
    printf("Thread %d has left the critical section\n", tid);
    fflush(stdout);
  }
  return NULL;
}

int main(int argc, char * argv[]) {
  pthread_t * threads;
  int * tids;

  assert(argc >= 2);
  nthreads = atoi(argv[1]);
  assert(nthreads>=0);
  threads = malloc(nthreads*sizeof(pthread_t));
  assert(threads);
  tids = malloc(nthreads*sizeof(int));
  assert(tids);
  for (int i=0; i<nthreads; i++)
    tids[i] = i;
  init_critical();
  for (int i=0; i<nthreads; i++)
    pthread_create(threads + i, NULL, thread, tids + i);
  for (int i=0; i<nthreads; i++)
    pthread_join(threads[i], NULL);
  finalize_critical();
  free(tids);
  free(threads);
}
