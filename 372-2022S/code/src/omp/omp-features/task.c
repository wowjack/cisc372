#include<stdio.h>

int main() {
  int x = 99;

#pragma omp parallel num_threads(2)
  {
#pragma omp for firstprivate(x)
    for (int i=0; i<20; i++) {
      printf("i=%d, x=%d\n", i, x);
      x=i;
    }
  }
}
