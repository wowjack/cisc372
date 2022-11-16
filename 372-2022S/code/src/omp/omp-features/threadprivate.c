#include <stdio.h>
#include <omp.h>

int x;

#pragma omp threadprivate(x)

void f() {
  // this updates the private copy of x...
  x=omp_get_thread_num();
}

int main() {
#pragma omp parallel num_threads(5)
  {
    int tid = omp_get_thread_num();

    f();
    printf("Thread %d: x = %d\n", tid, x);
  }
  printf("Final x = %d\n", x);
  return 0;
}
