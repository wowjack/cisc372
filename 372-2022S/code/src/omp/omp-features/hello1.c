#include <stdio.h>

int main () {
  printf("I am the master.\n"); // just the master
#pragma omp parallel
  {
    printf("Hello, world.\n"); // all threads
  } /* end of parallel region */
  printf("Goodbye, world.\n"); // just the master
}
