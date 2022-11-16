#include <stdio.h>
#include <omp.h>
int main() {
#pragma omp parallel
  {
#pragma omp for ordered
    for (int i=0; i<10; i++) {
      printf("Hello from %d\n", i);
#pragma omp ordered
      printf("Bye from %d\n", i);
    }
  }

}
