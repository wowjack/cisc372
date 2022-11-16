
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {
  assert(argc >= 2);
  int n = atoi(argv[1]);
  assert(n >= 0);
  int sum=0;
#pragma omp parallel for reduction(+:sum)
  for (int i=1; i<=n; i++)
    sum += i;
  printf("The sum is %d\n", sum);
}
