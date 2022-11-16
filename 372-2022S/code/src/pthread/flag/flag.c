/* Implementation of flag interface using Pthreads mutexes and
   condition variables. */
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include "flag.h"

void flag_init(flag_t * f, _Bool val) {
  f->val = val;
  pthread_mutex_init(&f->mutex, NULL);
  pthread_cond_init(&f->condition_var, NULL);
}

void flag_destroy(flag_t * f) {
  pthread_mutex_destroy(&f->mutex);
  pthread_cond_destroy(&f->condition_var);
}

void flag_raise(flag_t * f) {
  pthread_mutex_lock(&f->mutex);
  assert(!f->val);
  f->val = 1;
  pthread_cond_broadcast(&f->condition_var);
  pthread_mutex_unlock(&f->mutex);
}

void flag_lower(flag_t * f) {
  pthread_mutex_lock(&f->mutex);
  // wait until f->val != 0...
  while (!f->val) pthread_cond_wait(&f->condition_var, &f->mutex);
  f->val = 0;
  pthread_mutex_unlock(&f->mutex);
}
