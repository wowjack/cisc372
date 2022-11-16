#include <stdio.h>
#include <omp.h>
#define n 10
int a[n];
int main() {
#pragma omp parallel loop shared(a) default(none) num_threads(4)
  for (int i=0; i<n; i++)
    a[i] = i;
  for (int i=0; i<n; i++)
    printf("%d ", a[i]);
  printf("\n");
}
