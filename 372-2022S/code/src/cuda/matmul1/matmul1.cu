/* matmul1.cu:  simple CUDA version of matrix-matrix multiplication.
   Command line args are N, L, M.  A is NxL, B is LxM, C is NxM. */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

/* Number of threads per block in x and y direction. 32*32=1024 */
const int nthreadsx = 32, nthreadsy = 32;

double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

/* Allocate n*m doubles in the host's heap */
double * cpuAlloc(int n, int m) {
  double * result = (double*)malloc(n*m*sizeof(double));
  assert(result);
  return result;
}

/* Allocate n*m doubles on the device global memory */
double * gpuAlloc(int n, int m) {
  double * result;
  int err = cudaMalloc(&result, n*m*sizeof(double));
  assert(err == cudaSuccess);
  return result;
}

/* Print matrix mat, which has numRows rows and numCols cols */
void printMatrix(int numRows, int numCols, double * mat) {
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; j++)
      printf("%6.1f ", mat[i*numCols + j]);
    printf("\n");
  }
  printf("\n");
}

/* Kernel.  Multiplies a and b, sticking results into c.
   a is nxl, b is lxm, c is nxm. */
__global__ void multiply(int n, int l, int m,
			 double * a, double * b, double * c) {
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n && j < m) {
    double result = 0.0;
    for (int k = 0; k < l; k++)
      result += a[i*l + k] * b[k*m + j]; // a[i][k] * b[k][j];
    c[i*m + j] = result; // c[i][j]
  }
}

int main(int argc, char * argv[]) {
  assert(argc == 4);

  int err, N = atoi(argv[1]), L = atoi(argv[2]), M = atoi(argv[3]);

  assert(N>=1); assert(L>=1); assert(M>=1);

  dim3 blockDim(nthreadsx, nthreadsy),
    gridDim(M/nthreadsx + (M % nthreadsx != 0),
	    N/nthreadsy + (N % nthreadsy != 0));
  double * a = cpuAlloc(N, L), * b = cpuAlloc(L, M), * c = cpuAlloc(N, M),
    * a_d = gpuAlloc(N, L), * b_d = gpuAlloc(L, M), * c_d = gpuAlloc(N, M);

  printf("matmul1.cu: N=%d, L=%d, M=%d, gridDim=(%d,%d), blockDim=(%d,%d)\n",
	 N, L, M, gridDim.x, gridDim.y, blockDim.x, blockDim.y);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < L; j++)
      a[i*L + j] = rand()*1.0/RAND_MAX;
  for (int i = 0; i < L; i++)
    for (int j = 0; j < M; j++)
      b[i*M + j] = rand()*1.0/RAND_MAX;
#ifdef DEBUG
  printMatrix(N, L, a);
  printMatrix(L, M, b);
#endif
  printf("Starting computation.\n"); fflush(stdout);
  double time = mytime();
  err = cudaMemcpy(a_d, a, N*L*sizeof(double), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  err = cudaMemcpy(b_d, b, L*M*sizeof(double), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  multiply<<<gridDim,blockDim>>>(N, L, M, a_d, b_d, c_d);
  err = cudaMemcpy(c, c_d, N*M*sizeof(double), cudaMemcpyDeviceToHost);
  assert(err == cudaSuccess);
#ifdef DEBUG
  printMatrix(N, M, c);
#endif
  printf("Done.  Time = %lf.\n", mytime() - time);
}
