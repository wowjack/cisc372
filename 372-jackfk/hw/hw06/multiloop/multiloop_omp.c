#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char * argv[]) {
  long nrows, ncols, q, p;
  long double * rt, * ct;  
  
  assert(argc == 3);
  nrows = atoi(argv[1]);
  ncols = atoi(argv[2]);
  p = nrows / 2 - 1;
  q = ncols / 2 - 1;

  long double (*a)[ncols], (*b)[ncols];

  rt = (long double *)malloc(sizeof(long double) * nrows);
  ct = (long double *)malloc(sizeof(long double) * ncols);
  a = (long double (*)[ncols])malloc(sizeof(long double[ncols]) * nrows);
  b = (long double (*)[ncols])malloc(sizeof(long double[ncols]) * nrows);

  //This loop cannot be done in parallel with the next 2 because it writes to a and b and they read from a and b
  #pragma omp parallel for default(none) shared(nrows, ncols, a, b) collapse(2)
  for (int i = 0; i < nrows; i++)
    for (int j = 0; j < ncols; j++) {
      a[i][j] = sin(i + j);
      b[i][j] = cos(j);
    }

//these two loops are entirely separate so they can be done in parallel
#pragma omp parallel default(none) shared(nrows, ncols, p, q, rt, ct, a, b)
{
  #pragma omp for nowait schedule(dynamic, 3)
  for (int i = 0; i < nrows; i++) {
    rt[i] = 0.0;
    for (int j = 0; j < q; j++)
      rt[i] += a[i][j * 2] * a[i][j * 2 + 1];
  }
  #pragma omp for nowait schedule(dynamic, 3)
  for (int i = 0; i < ncols; i++) {
    ct[i] = 0.0;
    for (int j = 0; j < p; j++) 
      ct[i] += b[j * 2][i] * b[j * 2 + 1][i];
  }
  #pragma omp barrier
}
  //we want these to print out in order so we cannot parallelize these loops
  for (int i = 0; i < nrows; i++) 
    printf("rt[%d] = %6.4Lf\n", i, rt[i]);
  for (int i = 0; i < ncols; i++) 
    printf("ct[%d] = %6.4Lf\n", i, ct[i]);
  free(rt);
  free(ct);
  free(a);
  free(b);
}
