/* matmat.c: Matrix-matrix multiplication.
 */
#include <stdlib.h>
#include <stdio.h>

/* A is NxL, B is LxM, C is NxM. */
#define N 6
#define L 4
#define M 10

void printMatrix(int numRows, int numCols, double *m) {
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; j++)
      printf("%6.1f ", m[i*numCols + j]);
    printf("\n");
  }
  printf("\n");
}

/* Multiplies a vector and a matrix. */
void vecmat(double vector[L], double matrix[L][M], double result[M]) {
  for (int j = 0; j < M; j++) {
    result[j] = 0.0;
    for (int k = 0; k < L; k++)
      result[j] += vector[k]*matrix[k][j];
  }
}

int main() {
  double a[N][L], b[L][M], c[N][M];

  for (int i = 0; i < N; i++)
    for (int j = 0; j < L; j++)
      a[i][j] = i*L+j;
  for (int i = 0; i < L; i++)
    for (int j = 0; j < M; j++)
      b[i][j] = i*M+j;
  printMatrix(N, L, &a[0][0]);
  printMatrix(L, M, &b[0][0]);
  for (int i = 0; i < N; i++) vecmat(a[i], b, c[i]);
  printMatrix(N, M, &c[0][0]);
}
