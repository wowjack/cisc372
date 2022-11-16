/* Lamport's Bakery algorithm, in Pthreads */
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

int niter = 10;
int nthreads;
_Bool *entering;
pthread_mutex_t *entering_lock;
pthread_cond_t *entering_cond;
long * number;
pthread_mutex_t * number_lock;
pthread_cond_t * number_cond;

void init_critical() {
  entering = (_Bool*)malloc(nthreads*sizeof(_Bool));
  entering_lock = (pthread_mutex_t*)malloc(nthreads*sizeof(pthread_mutex_t));
  entering_cond = (pthread_cond_t*)malloc(nthreads*sizeof(pthread_cond_t));
  number = (long*)malloc(nthreads*sizeof(long));
  number_lock = (pthread_mutex_t*)malloc(nthreads*sizeof(pthread_mutex_t));
  number_cond = (pthread_cond_t*)malloc(nthreads*sizeof(pthread_cond_t));
}

void finalize_critical() {
  free(entering);
  free(entering_lock);
  free(entering_cond);
  free(number);
  free(number_lock);
  free(number_cond);
}

void enter_critical(int tid) {
  long mynum = -1;

  // Entering[i] = true;
  pthread_mutex_lock(entering_lock + tid);
  entering[tid] = true;
  pthread_mutex_unlock(entering_lock + tid);

  // Number[i] = 1 + max(Number[1], ..., Number[NUM_THREADS]);
  for (int j=0; j<nthreads; j++) {
    pthread_mutex_lock(number_lock + j);
    if (number[j] > mynum)
      mynum = number[j];
    pthread_mutex_unlock(number_lock + j);
  }
  mynum++;
  pthread_mutex_lock(number_lock + tid);
  number[tid] = mynum;
  pthread_cond_broadcast(number_cond + tid);
  pthread_mutex_unlock(number_lock + tid);
    
  // Entering[i] = false;
  pthread_mutex_lock(entering_lock + tid);
  entering[tid] = false;
  pthread_cond_broadcast(entering_cond + tid);
  pthread_mutex_unlock(entering_lock + tid);

  // for (integer j = 1; j <= NUM_THREADS; j++) {
  for (int j=0; j<nthreads; j++) {
    // while (Entering[j]) { /* nothing */ }
    pthread_mutex_lock(entering_lock + j);
    while (entering[j]) 
      pthread_cond_wait(entering_cond + j, entering_lock + j);
    pthread_mutex_unlock(entering_lock + j);

    // while ((Number[j] != 0) && ((Number[j], j) < (mynum, tid))) {}
    pthread_mutex_lock(number_lock + j);
    while (number[j] != 0 && (number[j]<mynum || (number[j]==mynum && j<tid)))
      pthread_cond_wait(number_cond + j, number_lock + j);
    pthread_mutex_unlock(number_lock + j);
  }
}

void exit_critical(int tid) {
  pthread_mutex_lock(number_lock + tid);
  number[tid] = 0;
  pthread_cond_broadcast(number_cond + tid);
  pthread_mutex_unlock(number_lock + tid);
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
  assert(argc >= 2);
  nthreads = atoi(argv[1]);
  assert(nthreads>=0);

  pthread_t threads[nthreads];
  int tids[nthreads];

  for (int i=0; i<nthreads; i++)
    tids[i] = i;
  init_critical();
  for (int i=0; i<nthreads; i++)
    pthread_create(threads + i, NULL, thread, tids + i);
  for (int i=0; i<nthreads; i++)
    pthread_join(threads[i], NULL);
  finalize_critical();
}
