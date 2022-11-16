#include <stdio.h>

int main() {
  int x = 5;

  printf("in sequential region, x = %d\n", x);
#pragma omp parallel default(none) firstprivate(x) num_threads(8)
  {
    printf("In parallel region, x = %d\n", x);
  }
}
