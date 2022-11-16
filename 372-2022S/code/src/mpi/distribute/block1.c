#include <stdio.h>
#ifndef N
#define N 20
#endif
unsigned int a[N];
int main() {
  unsigned long sum = 0;
  for (int i=0; i<N; i++)
    a[i] = i*i;
  for (int i=0; i<N; i++)
    sum += a[i];
  printf("sum = %ld\n", sum);
}
