#include <stdio.h>

int main() {

#pragma omp parallel
  {
    int x=99; // private
    int i;

    // wrong because x is not shared...
#pragma omp for firstprivate(x)
    for (i=0; i<10; i++) {
      x=x+i;
    } /* end of for */
    printf("x=%d\n", x);
  } /* end of parallel region */
}
