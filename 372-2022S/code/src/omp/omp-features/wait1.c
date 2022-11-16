/* Example using nowait correctly.  The two for loops can execute
   concurrently because they are accessing distinct variables. */
#include <stdio.h>
#include <stdlib.h>

#define n 100

int main () {
  double a[n], b[n];
#pragma omp parallel default(none) shared(a,b)
  {
#pragma omp for nowait
    for (int i=0; i<n; i++)
      a[i] = 2.0*i;
#pragma omp for
    for (int i=0; i<n; i++)
      b[i] = 3.0*i;
  } /* end of parallel region */
  for (int i=0; i<n; i++) {
    if (a[i] != 2.0*i) {
      printf("Error at a[%d]: %f\n", i, a[i]);
      fflush(stdout);
      exit(1);
    }
    if (b[i] !=3.0*i) {
      printf("Error at b[%d]: %f\n", i,b[i]);
      fflush(stdout);
      exit(1);
    }
  }
  printf("Success\n");
}
