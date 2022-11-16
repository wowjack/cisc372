/* matmul.c: Matrix-matrix multiplication.  Command line args are N,
   L, M.  A is NxL, B is LxM, C is NxM. */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

void printMatrix(int numRows, int numCols, double *m) {
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; j++)
      printf("%6.1f ", m[i*numCols + j]);
    printf("\n");
  }
  printf("\n");
}

double ** create2d(int n, int m) {
  double * storage = malloc(n*m*sizeof(double));
  assert(storage);
  double ** rows = malloc(n*sizeof(double*));
  assert(rows);
  for (int i=0; i<n; i++)
    rows[i] = &storage[i*m];
  return rows;
}

int main(int argc, char * argv[]) {
  assert(argc == 4);
  int N = atoi(argv[1]), L = atoi(argv[2]), M = atoi(argv[3]);
  assert(N>=1);
  assert(L>=1);
  assert(M>=1);
  double ** a = create2d(N, L), ** b = create2d(L, M), ** c = create2d(N,M);

  printf("matmul: N=%d, L=%d, M=%d\n", N, L, M);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < L; j++)
      a[i][j] = rand()*1.0/RAND_MAX;
  for (int i = 0; i < L; i++)
    for (int j = 0; j < M; j++)
      b[i][j] = rand()*1.0/RAND_MAX;
#ifdef DEBUG
  printMatrix(N, L, &a[0][0]);
  printMatrix(L, M, &b[0][0]);
#endif
  printf("Starting computation.\n"); fflush(stdout);
  double time = mytime();
  for (int i = 0; i < N; i++)
    for (int j = 0; j < M; j++)
      c[i][j] = 0.0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < L; k++)
	c[i][j] += a[i][k]*b[k][j];
    }
  }
#ifdef DEBUG
  printMatrix(N, M, &c[0][0]);
#endif
  printf("Done.  Time = %lf.\n", mytime() - time);
}
