/* Add up some numbers in Pthreads.
 * This time race conditions avoided by using a mutex.
 * The number of threads to create is specified as the command line argument.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>

int nthreads; // number of threads to created
int sum = 0;
pthread_mutex_t mutexsum;


/* The thread function.
 * The signature must be void* -> void*
 */
void* hello(void* arg) {
  // always ok to cast void* to T* for any type T...
  int * tidp = (int*)arg; 

  pthread_mutex_lock (&mutexsum);
  sum += (*tidp)+1;
  pthread_mutex_unlock (&mutexsum);
  return NULL;
}

int main (int argc, char *argv[]) {
  assert(argc >= 2);
  nthreads = atoi(argv[1]);
  assert(nthreads>=0);

  pthread_t threads[nthreads];
  int tids[nthreads];
  
  pthread_mutex_init(&mutexsum, NULL);
  for (int i=0; i<nthreads; i++)
    tids[i] = i;
  for (int i=0; i<nthreads; i++)
    pthread_create(threads + i, NULL, hello, tids + i);
  for (int i=0; i<nthreads; i++)
    pthread_join(threads[i], NULL);
  pthread_mutex_destroy(&mutexsum);
  printf("The sum is %d\n", sum);
}
