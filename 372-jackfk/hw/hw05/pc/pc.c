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

//Threads will gain lock on bufSizeMutex when reading or writing to the buffer size variable
//Threads will gain lock on firstEleMutex when reading or writing to the buffer first element variable
pthread_mutex_t bufSizeMutex, firstEleMutex;
//Consumer will wait on bufEmptyCondVar when the buffer is empty
//Producer will wait on bufFullCondVar when the buffer is full
pthread_cond_t bufEmptyCondVar, bufFullCondVar;


double tmp = 0.0; // just used for fake busy work

/* Repeatedly generates numbers and puts them in the buffer.  ptr is
   ignored. */
void * producer (void * ptr) {
  int num;

  do {
    for (int i=0; i<100; i++) num = rand()%1002 - 1; // do some work
    pthread_mutex_lock(&bufSizeMutex);
    while(size>=cap){
      pthread_cond_wait(&bufFullCondVar, &bufSizeMutex);
    }
    pthread_mutex_lock(&firstEleMutex);
    //printf("Producer: inserting into buffer...%4d\n", num); fflush(stdout);
    buf[(first+size)%cap] = num;
    size++;
    if(size==1){ //If the buffer previously had zero items, signal the consumer to consume
      pthread_cond_signal(&bufEmptyCondVar);
    }
    pthread_mutex_unlock(&bufSizeMutex);
    pthread_mutex_unlock(&firstEleMutex);
  } while (num != -1);
  return NULL;
}

/* Repeatedly consumes from buffer and accumulates the sum.  ptr is a
   pointer to where the final sum should be stored. */
void * consumer (void * ptr) {
  int num, s=0;
  
  while (1) {
    // here I should wait for buf to be non-empty
    pthread_mutex_lock(&bufSizeMutex);
    while(size<=0){
      pthread_cond_wait(&bufEmptyCondVar, &bufSizeMutex);
    }
    pthread_mutex_lock(&firstEleMutex);
    num = buf[first];
    //printf("Consumer: removing from buffer...........%4d\n", num); fflush(stdout);
    first = (first+1)%cap;
    size--;
    if(size==cap-1){ //If buffer size is less than cap, signal the producer to produce
      pthread_cond_signal(&bufFullCondVar);
    }
    pthread_mutex_unlock(&bufSizeMutex);
    pthread_mutex_unlock(&firstEleMutex);
    if (num == -1) break;
    for (int i=0; i<200; i++) tmp+=sin(i); // do some work
    s += num;
  }
  *((int*)ptr) = s;
  return NULL;
}

int main(int argc, char *argv[]) {
  //initialize cond vars and mutexes
  pthread_mutex_init(&bufSizeMutex, NULL); pthread_mutex_init(&firstEleMutex, NULL);
  pthread_cond_init(&bufEmptyCondVar, NULL); pthread_cond_init(&bufFullCondVar, NULL);

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

  pthread_mutex_destroy(&bufSizeMutex); pthread_mutex_destroy(&firstEleMutex);
  pthread_cond_destroy(&bufEmptyCondVar); pthread_cond_destroy(&bufFullCondVar);
}
