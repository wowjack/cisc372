/* matmat_mpi.c: MPI version of matrix-matrix multiplication, using
   manager-worker pattern.  A task involves multiplying one row of
   matrix A by the entire matrix B to get one row of the result, C.

   Based on program from Using MPI, modified by S.F.Siegel. 
*/
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

/* A is NxL, B is LxM, C is NxM. */
#define N 6
#define L 4
#define M 10
#define comm MPI_COMM_WORLD

int rank, nprocs;

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

/* Proc 0: distributes tasks to workers, accumulates the results */
void manager() {
  MPI_Status status;
  double a[N][L], b[L][M], c[N][M], tmp[M];
  int count; // number of tasks sent out
  int worker_counts[nprocs]; // number completed by each worker
  
  for (int i = 0; i < nprocs; i++)
    worker_counts[i] = 0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < L; j++)
      a[i][j] = i*L+j;
  for (int i = 0; i < L; i++)
    for (int j = 0; j < M; j++)
      b[i][j] = i*M+j;
  printMatrix(N, L, &a[0][0]);
  printMatrix(L, M, &b[0][0]);
  // Broadcast entire matrix B to all workers...
  MPI_Bcast(&b[0][0], L*M, MPI_DOUBLE, 0, comm);
  // Send one task to each worker, unless you run out of tasks...
  for (count = 0; count < nprocs-1 && count < N; count++)
    MPI_Send(&a[count][0], L, MPI_DOUBLE, count+1, count+1, comm);
  // Receive result, insert into C, send the next task, repeat...
  for (int i = 0; i < N; i++) {
    MPI_Recv(tmp, M, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
    for (int j = 0; j < M; j++)
      c[status.MPI_TAG-1][j] = tmp[j];
    worker_counts[status.MPI_SOURCE]++;
    if (count < N) {
      MPI_Send(&a[count][0], L, MPI_DOUBLE, status.MPI_SOURCE, count+1, comm);
      count++;
    }
  }
  // send termination signals (tag=0) to all workers...
  for (int i = 1; i < nprocs; i++)
    MPI_Send(NULL, 0, MPI_INT, i, 0, comm);
  printMatrix(N, M, &c[0][0]);
  for (int i = 1; i < nprocs; i++)
    printf("Worker %d completed %d tasks.\n", i, worker_counts[i]);
}

/* Worker: multiplies a vector and a matrix, repeats */
void worker() {
  double b[L][M], in[L], out[M];
  MPI_Status status;

  MPI_Bcast(&b[0][0], L*M, MPI_DOUBLE, 0, comm);
  while (1) {
    MPI_Recv(in, L, MPI_DOUBLE, 0, MPI_ANY_TAG, comm, &status);
    if (status.MPI_TAG == 0) break;
    vecmat(in, b, out);
    MPI_Send(out, M, MPI_DOUBLE, 0, status.MPI_TAG, comm);
  }
}

int main() {
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);
  if (rank == 0) manager(); else worker();
  MPI_Finalize();
}
