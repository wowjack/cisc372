#include <omp.h>
#include <stdio.h>

unsigned int s = 0;

int main() {
  int tid;

#pragma omp parallel num_threads(10)
  {
    int i;
    tid = omp_get_thread_num();

    // inserted for slowness...
    for (i=0; i<1000; i++)
#pragma omp critical
      s+= i;
    printf("Thread %d: tid = %d\n", omp_get_thread_num(), tid);
  }
  printf("s=%u\n", s);
}
