/* Fix the synchronization! */
#include <stdio.h>
#include <pthread.h>
#define N 10

int s1 = 0, s2 = 0;

void * f1(void * arg) {
  for (int i=1; i<=N; i++) {
    printf("Thread 1: waiting for signal.\n");
    fflush(stdout);
    // wait for s1 >= i
    printf("Thread 1: signal received: s1=%d.\n", s1);
    fflush(stdout);
    s2 = i;
    printf("Thread 1: sending signal.\n");
    fflush(stdout);
  }
  printf("Thread 1: terminating.\n");
  fflush(stdout);
  return NULL;
}

void * f2(void * arg) {
  for (int i=1; i<=N; i++) {
    printf("\tThread 2: sending signal.\n");
    fflush(stdout);
    s1 = i;
    printf("\tThread 2: waiting for signal.\n");
    fflush(stdout);
    // wait for s2 >= i
    printf("\tThread 2: signal received: s2=%d.\n", s2);
    fflush(stdout);
  }
  printf("\tThread 2: terminating.\n");
  fflush(stdout);
  return NULL;
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t2, NULL, f2, NULL);
  pthread_create(&t1, NULL, f1, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
}
