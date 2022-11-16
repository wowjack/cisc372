
#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>

int main () {
  unsigned int i = 0;
  unsigned long long sum = 0ULL;
  unsigned long long N = 3000000000ULL;

  printf("long long max = %llu\n", ULLONG_MAX);
  printf("expected result = %llu\n", N*(N-1)/2);
  fflush(stdout);
#pragma omp parallel shared(sum,N) private(i)
  {
    unsigned long long psum = 0ULL;

    printf("psum=%llu\n", psum);
    fflush(stdout);
#pragma omp for
    for (i=0; i<N; i++) {
      psum += i;
    }
#pragma omp critical
    sum += psum;
  }
  printf("sum=%llu\n", sum);
  fflush(stdout);  
  assert(sum == N*(N-1)/2);
}
