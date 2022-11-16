#include <stdio.h>
#include <omp.h>
#define n 10
int a[n], b[n];
int main() {
  for (int i=0; i<n; i++)
    a[i] = i;
#pragma omp parallel for shared(a,b) default(none) num_threads(4)
  for (int i=0; i<n-1; i++)
    b[i] = a[i+1] - a[i];
  for (int i=0; i<n-1; i++)
    printf("%d ", b[i]);
  printf("\n");
}
