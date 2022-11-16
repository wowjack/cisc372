/* Incorrect synchronization.  Use mutexes and condition variables
 * to fix. */
#include <stdio.h>
#include <pthread.h>

int s = 0;

void * f1(void * arg) {
  printf("Thread 1: waiting for signal.\n");
  fflush(stdout);
  // wait for s!=0 here
  printf("Thread 1: signal received: s=%d\n", s);
  fflush(stdout);
  return NULL;
}

void * f2(void * arg) {
  printf("\tThread 2: sending signal.\n");
  fflush(stdout);
  s = 1;
  return NULL;
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, f1, NULL);
  pthread_create(&t2, NULL, f2, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
}
