#include <omp.h>
#include <stdio.h>

int main() {
  int nthreads, tid;

#pragma omp parallel private(nthreads, tid) num_threads(10)
  {
    tid = omp_get_thread_num();
    printf("Hello World from thread = %d\n", tid);
    if (tid == 0) { // only master
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
  } // end of parallel region
}
