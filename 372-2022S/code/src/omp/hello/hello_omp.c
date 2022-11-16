/* Hello world in OpenMP.  One command line arg: number of threads you
   are requesting.  That might be different than the number you
   actually get.  */
#include <assert.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

int main (int argc, char *argv[]) {
  int request;

  assert(argc==2);
  request = atoi(argv[1]);
#pragma omp parallel num_threads(request) // start parallel region
  {
    int nthreads = omp_get_num_threads(); // number of threads in this region
    int tid = omp_get_thread_num(); // ID number of this tread
    
    printf("Hello from OpenMP thread %d of %d!\n", tid, nthreads);
  } // end of parallel region
  printf("Thread %d is still here.\n", omp_get_thread_num());
}
