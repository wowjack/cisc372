/* "Hello, world" in Pthreads.
 * The number of threads to create is specified as the command line argument.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>

int nthreads; // number of threads to created
int mthreads;

/* The thread function.
 * The signature must be void* -> void*
 */
void* hello(void* arg) {
  // always ok to cast from T* to void* and back, for any object type T...
  int * tidp = (int*)arg; 

  printf("Hello from Pthread thread %d of %d!\n", *tidp, nthreads*mthreads);
  return NULL;
}

int main (int argc, char *argv[]) {
  assert(argc==3);
  nthreads = atoi(argv[1]);
  mthreads = atoi(argv[2]);

  assert(nthreads>=0);

  pthread_t threads[nthreads * mthreads];
  int tids[nthreads * mthreads];
  
  for (int i=0; i<nthreads*mthreads; i++)
    tids[i] = i;
  for (int i=0; i<nthreads; i++){
    for(int j=0; j<mthreads; j++){
      pthread_create(threads+(i*j)+j, NULL, hello, tids+(i*j)+j);
    }
  }
  printf("Hello from the main thread\n");
  for (int i=0; i<nthreads*mthreads; i++)
    pthread_join(threads[i], NULL);
}
