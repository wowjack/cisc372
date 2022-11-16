/* A basic semaphore library, using Pthreads mutexes and condition variables.
 * Author: Stephen F. Siegel
 * Date: Oct 2018
 */
#include "sem.h"
#include <stdlib.h>
#include <pthread.h>

void semaphore_init(semaphore *s, int val) {
  s->val = val;
  pthread_mutex_init(&s->mutex, NULL);
  pthread_cond_init(&s->condition_var, NULL);
}

void semaphore_destroy(semaphore *s) {
  pthread_mutex_destroy(&s->mutex);
  pthread_cond_destroy(&s->condition_var);
}

void semaphore_V(semaphore *s) {
  pthread_mutex_lock(&s->mutex);
  ++(s->val);
  pthread_cond_broadcast(&s->condition_var);
  pthread_mutex_unlock(&s->mutex);
}

void semaphore_P(semaphore *s) {
  pthread_mutex_lock(&s->mutex);
  // wait until s->val > 0...
  while (s->val == 0) pthread_cond_wait(&s->condition_var, &s->mutex);
  --(s->val);
  pthread_mutex_unlock(&s->mutex);
}
