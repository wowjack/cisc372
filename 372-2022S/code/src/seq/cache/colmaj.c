#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#define N 20000
double ** a, * x, * y;

static double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

static double ** allocate2d(int n, int m) {
  double * storage = malloc( n * m * sizeof(double) );
  double ** a = malloc( n * sizeof(double*) );
  
  assert(storage);
  assert(a);
  for (int i=0; i<n; i++) a[i] = & storage[ i * m ];
  return a;
}

static void free2d(double ** a) {
  free(a[0]); // frees storage
  free(a);    // frees a
}

int main() {
  a = allocate2d(N, N);
  x = malloc(N*sizeof(double));
  assert(x);
  y = malloc(N*sizeof(double));
  assert(y);
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      a[i][j] = (i*N+j)/1000;
  for (int i=0; i<N; i++)
    x[i] = i;
  for (int i=0; i<N; i++)
    y[i] = 0.0;
  printf("Starting computation.\n");
  fflush(stdout);
  double t0 = mytime();
  for (int j=0; j<N; j++)
    for (int i=0; i<N; i++)
      y[i] += a[i][j]*x[j];
  printf("Computation complete. Time = %lf seconds\n", mytime()-t0);
  free(x);
  free(y);
  free2d(a);
}
