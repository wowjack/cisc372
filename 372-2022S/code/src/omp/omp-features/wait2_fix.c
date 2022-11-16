/* wait2_fix.c: fixes the data race in wait2.c by removing the
   "nowait". */
#include <stdio.h>
#include <stdlib.h>

#define n 100000

int main() {
  double a[n], b[n];
#pragma omp parallel default(none) shared(a,b)
  {
#pragma omp for
    for (int i=0; i<n; i++)
      a[i] = 2.0*i;
#pragma omp for
    for (int i=0; i<n; i++)
      b[i] = 2.0*a[n-i-1];
  } /* end of parallel region */
  for (int i=0; i<n; i++) {
    if (a[i] != 2.0*i) {
      printf("Error at a[%d]: %f\n", i, a[i]);
      fflush(stdout);
      exit(1);
    }
    if (b[i] !=2.0*(2.0*(n-i-1))) {
      printf("Error at b[%d]: %f\n", i,b[i]);
      fflush(stdout);
      exit(1);
    }
  }
  printf("Success 2\n");
}
