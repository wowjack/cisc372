#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

double ** create_2d(int n, int m) {
  double * storage = malloc(n*m*sizeof(double));
  assert(storage);
  double ** rows = malloc(n*sizeof(double*));
  assert(rows);
  for (int i=0; i<n; i++)
    rows[i] = &storage[i*m];
  return rows;
}

void destroy_2d(double ** a) {
  free(a[0]); // free storage
  free(a); // free rows
}

int main(int argc, char * argv[]) {
  int n = atoi(argv[1]), m = atoi(argv[2]);
  double ** a = create_2d(n, m);
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      a[i][j] = 100*i + j;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++)
      printf("%7.1lf ", a[i][j]);
    printf("\n");
  }
  destroy_2d(a);
}
