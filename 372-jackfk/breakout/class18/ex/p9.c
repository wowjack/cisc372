#include <stdio.h>
#include <omp.h>
#define n 10
int a[n], b[n];
int main() {
  for (int i=0; i<n; i++)
    a[i] = b[i] = i;
#pragma omp parallel loop shared(a,b) default(none) num_threads(4)
  for (int i=0; i<n-1; i++) {
    int tmp = 2*b[i+1];
    int x = a[i]/2;
    b[i] = x*tmp;
  }
  for (int i=0; i<n-1; i++)
    printf("%d ", b[i]);
  printf("\n");
}
