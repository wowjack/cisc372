/* A two-process barrier using two flags */
#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include "flag.h"

pthread_t t1, t2;
pthread_mutex_t mutex;
flag_t s1, s2;
int i1=0, i2=0;

const int N=10000;

void *f1(void* arg) {
  while (i1<N) {
    pthread_mutex_lock(&mutex);
    i1++;
    pthread_mutex_unlock(&mutex);
    flag_raise(&s1);
    flag_lower(&s2);
    pthread_mutex_lock(&mutex);
    assert(i1==i2);
    pthread_mutex_unlock(&mutex);
    flag_raise(&s1);
    flag_lower(&s2);
  }
  return NULL;
}

void *f2(void* arg) {
  while (i2<N) {
    pthread_mutex_lock(&mutex);
    i2++;
    pthread_mutex_unlock(&mutex);
    flag_raise(&s2);
    flag_lower(&s1);
    pthread_mutex_lock(&mutex);
    assert(i1==i2);
    pthread_mutex_unlock(&mutex);
    flag_raise(&s2);
    flag_lower(&s1);
  }
  return NULL;
}


int main() {
  pthread_mutex_init(&mutex, NULL);
  flag_init(&s1, 0);
  flag_init(&s2, 0);
  pthread_create(&t1, NULL, f1, NULL);
  pthread_create(&t2, NULL, f2, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  flag_destroy(&s1);
  flag_destroy(&s2);
  pthread_mutex_destroy(&mutex);
  printf("Barrier operated correctly.\n");
}
