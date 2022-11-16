#include <stdio.h>
#include <omp.h>

int main() {
  int x = 10, y=37;
#pragma omp parallel firstprivate(x) shared(y) default(none) num_threads(4)
  {
    int tid = omp_get_thread_num();
    x += tid;
    printf("Thread %d: x=%d\n", tid, x);
    if (tid == 1) {
      y=17;
      printf("Thread %d: y=%d\n", tid, y);
    }
  }
  printf("x=%d, y=%d\n", x, y);
}
