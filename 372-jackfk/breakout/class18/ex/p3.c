#include <stdio.h>
#include <omp.h>

int main() {
  int x = 10, y=37;
#pragma omp parallel private(x) shared(y) default(none) num_threads(4)
  {
    int tid = omp_get_thread_num();
    x = y = tid;
    printf("Thread %d: x=%d, y=%d\n", tid, x, y);
  }
  printf("x=%d, y=%d\n", x, y);
}
