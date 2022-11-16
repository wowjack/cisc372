/* Producer-consumer example. Single producer, single consumer, single datum.
 *
 * Author: Stephen F. Siegel
 * Date : 2016-oct-24
 */
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "flag.h"

int niter = 100; // number of times to loop
flag_t f1, f2; // signals to read and write
int buf; // the buffer

void * producer(void * arg) {
  int value; // local var to store value produced

  for (int i=0; i<niter; i++) {
    value = rand()%100; // represents "producing" some value
    flag_lower(&f1); // wait for signal from consumer: safe to write
    buf = value;
    printf("Producer: produced %d\n", value);
    fflush(stdout);
    flag_raise(&f2); // signal to consumer: safe to read
  }
  return NULL;
}

void * consumer(void * arg) {
  int value; // local var to store value consumed
  
  for (int i=0; i<niter; i++) {
    flag_lower(&f2); // wait for signal from producer: safe to read
    value = buf;
    printf("Consumer: read %d\n", value);
    fflush(stdout);
    flag_raise(&f1); // signal to producer: safe to write
  }
  return NULL;
}

int main() {
  flag_init(&f1, 1);
  flag_init(&f2, 0);
  
  pthread_t producer_thread, consumer_thread;

  pthread_create(&producer_thread, NULL, producer, NULL);
  pthread_create(&consumer_thread, NULL, consumer, NULL);
  pthread_join(producer_thread, NULL);
  pthread_join(consumer_thread, NULL);
  flag_destroy(&f1);
  flag_destroy(&f2);
}
