/* sections.c: Example of sections construct */
#include <stdio.h>
#include <omp.h>
#include <limits.h>
#define N 20
typedef unsigned long ulong;

ulong sumUpTo(int n) {
  ulong s=0;
  for (int i=1; i<=n; i++) s+=i;
  return s;
}

ulong productUpTo(int n) {
  ulong p=1;
  for (int i=1; i<=n; i++) p*=i;
  return p;
}

int main() {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    if (tid == 0) printf("Number of threads: %d\n", omp_get_num_threads());
#pragma omp sections
    {
#pragma omp section
      {
	printf("Thread %d: sum to %d ........... %lu\n", tid, N, sumUpTo(N));
      }
#pragma omp section
      {
	printf("Thread %d: product to %d ....... %lu\n", tid, N, productUpTo(N));
      }
    } /* end of sections */
  } /* end of parallel region */
}
