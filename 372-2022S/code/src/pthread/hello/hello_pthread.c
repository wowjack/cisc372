/* "Hello, world" in Pthreads.
 * The number of threads to create is specified as the command line argument.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>

int nthreads; // number of threads to created

/* The thread function.
 * The signature must be void* -> void*
 */
void* hello(void* arg) {
  // always ok to cast from T* to void* and back, for any object type T...
  int * tidp = (int*)arg; 

  printf("Hello from Pthread thread %d of %d!\n", *tidp, nthreads);
  return NULL;
}

int main (int argc, char *argv[]) {
  assert(argc==2);
  nthreads = atoi(argv[1]);
  assert(nthreads>=0);

  pthread_t threads[nthreads];
  int tids[nthreads];
  
  for (int i=0; i<nthreads; i++)
    tids[i] = i;
  for (int i=0; i<nthreads; i++)
    pthread_create(threads + i, NULL, hello, tids + i);
  printf("Hello from the main thread\n");
  for (int i=0; i<nthreads; i++)
    pthread_join(threads[i], NULL);
}
