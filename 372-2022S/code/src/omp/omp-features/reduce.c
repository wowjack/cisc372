#include <stdio.h>
#include <omp.h>

#define n 10

int a[n];
int s=1000000;

int main() {
  printf("Start s = %d\n", s);
  fflush(stdout);
#pragma omp parallel default(none) shared(a,s)
  {
    int i, tid = omp_get_thread_num();

#pragma omp for
    for (i=0; i<n; i++) {
      a[i] = i;
    }
#pragma omp for reduction(+:s) schedule(static,1)
    for (i=0; i<n; i++) {
      s+=a[i];
      printf("Local s on thread %d = %d\n", tid, s);
    }
  }
  printf("Final s = %d\n", s);
  return 0;
}
