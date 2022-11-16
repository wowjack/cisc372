/* This example illustrates that the private copy of x is unitialized
   upon entering the parallel region */
#include <stdio.h>

int main() {
  int x = 5;

  printf("in sequential region, x = %d\n", x);
#pragma omp parallel default(none) private(x) num_threads(8)
  {
    printf("In parallel region, x = %d\n", x);
  }
  printf("Back in sequential region, x = %d\n", x);
}
