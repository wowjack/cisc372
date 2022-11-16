/* Incorrect synchronization.  Use mutexes and condition variables
 * to fix. */
#include <stdio.h>
#include <pthread.h>

int s = 0;

//My code
pthread_cond_t condVar;
pthread_mutex_t mutex;
//End my code

void * f1(void * arg) {
  printf("Thread 1: waiting for signal.\n");
  fflush(stdout);
  // wait for s!=0 here

  //My code
  pthread_mutex_lock(&mutex);
  while(s == 0){
    pthread_cond_wait(&condVar, &mutex);
  }
  //End my code

  printf("Thread 1: signal received: s=%d\n", s);
  fflush(stdout);
  return NULL;
}

void * f2(void * arg) {
  printf("\tThread 2: sending signal.\n");
  fflush(stdout);

  //My code
  pthread_mutex_lock(&mutex);
  s = 1;
  //Does it matter in which order you unlock the mutex and broadcast/signal?
  //I tried each a few times and nothing broke so idk
  pthread_mutex_unlock(&mutex);
  pthread_cond_broadcast(&condVar);
  //End my code

  return NULL;
}

int main() {
  //My code
  pthread_cond_init(&condVar, NULL);
  pthread_mutex_init(&mutex, NULL);
  //End my code

  pthread_t t1, t2;
  pthread_create(&t1, NULL, f1, NULL);
  pthread_create(&t2, NULL, f2, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
}
