/* A two-process barrier using two flags */
#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include "flag.h"

pthread_t t1, t2;
pthread_mutex_t mutex;
flag_t f1, f2;
int i1=0, i2=0;

const int N=10000;

void * thread1(void * arg) {
  while (i1<N) {
    pthread_mutex_lock(&mutex);
    i1++;
    pthread_mutex_unlock(&mutex);
    flag_raise(&f1);
    flag_lower(&f2);
    pthread_mutex_lock(&mutex);
    assert(i1==i2);
    pthread_mutex_unlock(&mutex);
    flag_raise(&f1);
    flag_lower(&f2);
  }
  return NULL;
}

void * thread2(void * arg) {
  while (i2<N) {
    pthread_mutex_lock(&mutex);
    i2++;
    pthread_mutex_unlock(&mutex);
    flag_lower(&f1);
    flag_raise(&f2);
    pthread_mutex_lock(&mutex);
    assert(i1==i2);
    pthread_mutex_unlock(&mutex);
    flag_lower(&f1);
    flag_raise(&f2);
  }
  return NULL;
}

int main() {
  pthread_mutex_init(&mutex, NULL);
  flag_init(&f1, 0);
  flag_init(&f2, 0);
  pthread_create(&t1, NULL, thread1, NULL);
  pthread_create(&t2, NULL, thread2, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  flag_destroy(&f1);
  flag_destroy(&f2);
  pthread_mutex_destroy(&mutex);
  printf("Barrier operated correctly.\n");
}
