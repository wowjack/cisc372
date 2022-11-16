/* Simple solution to critical section problem using mutexes */

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

int niter = 10;
int nthreads;
pthread_mutex_t mutex;

void init_critical() {
  pthread_mutex_init(&mutex, NULL);
}

void finalize_critical() {
  pthread_mutex_destroy(&mutex);
}

void enter_critical(int tid) {
  pthread_mutex_lock(&mutex);
}

void exit_critical(int tid) {
  pthread_mutex_unlock(&mutex);
}

void * thread(void * arg) {
  int tid = *((int*)arg);
  
  for (int i=0; i< niter; i++) {
    enter_critical(tid);
    printf("Thread %d is in the critical section\n", tid);
    fflush(stdout);
    printf("Thread %d is leaving the critical section\n", tid);
    fflush(stdout);
    exit_critical(tid);
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
