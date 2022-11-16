/* Producer-consumer with bounded buffer.  Uncomment printfs to see
   more output.  */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/* This is a bounded buffer (or "circular buffer") for a FIFO queue.
   first is index of first (oldest) message in the queue.  size is the
   number of messages currently in the queue.  cap is the capacity
   (maximum size) of the queue. buf is an array of length cap. */
int * buf, first, size, cap;

double tmp = 0.0; // just used for fake busy work

/* Repeatedly generates numbers and puts them in the buffer.  ptr is
   ignored. */
void * producer (void * ptr) {
  int num;

  do {
    for (int i=0; i<100; i++) num = rand()%1002 - 1; // do some work
    // here I should wait for buf to be non-full
    while (size==cap); // this is not the right way!
    //printf("Producer: inserting into buffer...%4d\n", num); fflush(stdout);
    buf[(first+size)%cap] = num;
    size++;
  } while (num != -1);
  return NULL;
}

/* Repeatedly consumes from buffer and accumulates the sum.  ptr is a
   pointer to where the final sum should be stored. */
void * consumer (void * ptr) {
  int num, s=0;
  
  while (1) {
    // here I should wait for buf to be non-empty
    while (size==0); // this is not the right way!
    num = buf[first];
    //printf("Consumer: removing from buffer...........%4d\n", num); fflush(stdout);
    first = (first+1)%cap;
    size--;
    if (num == -1) break;
    for (int i=0; i<200; i++) tmp+=sin(i); // do some work
    s += num;
  }
  *((int*)ptr) = s;
  return NULL;
}

int main(int argc, char *argv[]) {
  int sum;
  
  if (argc != 2) {
    printf("Provide one argument: capacity of buffer.\n");
    exit(1);
  }
  cap = atoi(argv[1]);
  buf = malloc(cap*sizeof(int));
  assert(buf);
  first = size = 0;
  pthread_t t0, t1;
  pthread_create(&t0, NULL, producer, NULL);
  pthread_create(&t1, NULL, consumer, (void*)&sum);
  pthread_join(t0, NULL);
  pthread_join(t1, NULL);
  free(buf);
  printf("Sum: %d\n", sum);
}
