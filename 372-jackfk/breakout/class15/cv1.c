/* Fix the synchronization! */
#include <stdio.h>
#include <pthread.h>
#define N 10

int s1 = 0, s2 = 0;

pthread_cond_t cond1, cond2;
pthread_mutex_t mutex1, mutex2;

void * f1(void * arg) {
  for (int i=1; i<=N; i++) {
    printf("Thread 1: waiting for signal.\n");
    fflush(stdout);

    // wait for s1 >= i
    pthread_mutex_lock(&mutex1);
    while(s1<i){
      pthread_cond_wait(&cond1, &mutex1);
    }
    pthread_mutex_unlock(&mutex1);

    printf("Thread 1: signal received: s1=%d.\n", s1);
    fflush(stdout);

    pthread_mutex_lock(&mutex2);
    s2 = i;
    pthread_mutex_unlock(&mutex2);
    pthread_cond_broadcast(&cond2);

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

    pthread_mutex_lock(&mutex1);
    s1 = i;
    pthread_mutex_unlock(&mutex1);
    pthread_cond_broadcast(&cond1);

    printf("\tThread 2: waiting for signal.\n");
    fflush(stdout);

    // wait for s2 >= i
    pthread_mutex_lock(&mutex2);
    while(s2<i){
      pthread_cond_wait(&cond2, &mutex2);
    }
    pthread_mutex_unlock(&mutex2);

    printf("\tThread 2: signal received: s2=%d.\n", s2);
    fflush(stdout);
  }
  printf("\tThread 2: terminating.\n");
  fflush(stdout);
  return NULL;
}

int main() {

  pthread_mutex_init(&mutex1, NULL);
  pthread_mutex_init(&mutex2, NULL);
  pthread_cond_init(&cond1, NULL);
  pthread_cond_init(&cond2, NULL);

  pthread_t t1, t2;
  pthread_create(&t2, NULL, f2, NULL);
  pthread_create(&t1, NULL, f1, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
}
