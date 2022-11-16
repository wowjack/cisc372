/* Add up some numbers in Pthreads.  Watch out for a DATA RACE!!!!
 * The number of threads to create is specified as the command line argument.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>

int nthreads; // number of threads to create
int sum = 0;

/* The thread function.
 * The signature must be void* -> void*
 */
void* hello(void* arg) {
  int * tidp = (int*)arg; 

  sum += (*tidp)+1;
  return NULL;
}

int main(int argc, char *argv[]) {
  assert(argc >= 2);
  nthreads = atoi(argv[1]);
  assert(nthreads>=0);

  pthread_t threads[nthreads];
  int tids[nthreads];
  
  for (int i=0; i<nthreads; i++)
    tids[i] = i;
  for (int i=0; i<nthreads; i++)
    pthread_create(threads + i, NULL, hello, tids + i);
  for (int i=0; i<nthreads; i++)
    pthread_join(threads[i], NULL);
  printf("The sum is %d\n", sum);
}
