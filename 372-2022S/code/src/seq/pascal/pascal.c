#include <stdio.h>

#ifndef N
#define N 20
#endif

unsigned int a1[2*N+1], a2[2*N+1];
unsigned int *p=a1, *q=a2;

void print() {
  for (int j=0; j<2*N+1; j++) {
    if (p[j]==0)
      printf("      ");
    else 
      printf("%6u", p[j]);
  }
  printf("\n");
}

/* When this is called p holds the current values, and when it returns,
 * p will hold the next set of values. */
void update() {
  q[0] = p[1];
  for (int j=1; j<2*N; j++)
    q[j] = p[j-1]+p[j+1];
  q[2*N] = p[2*N-1];
  unsigned int *tmp = p; p=q; q=tmp; // swap pointers
}

int main() {
  p[N] = 1;
  for (int i=0; i<N; i++) {
    print();
    update();
  }
  print();
}
