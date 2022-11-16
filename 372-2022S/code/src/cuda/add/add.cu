#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>

/* gridDim.x   : # blocks
   blockIdx.x  : ID number of this block
   blockDim.x  : # threads per block
   threadIdx.x : local ID number of this thread within the block

   blockDim.x * blockIdx.x + threadIdx.x : global thread ID number
 */

const int nblocks = 45;
const int threadsPerBlock = 128;

double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

__global__ void vec_add(int n, double * a, double * b, double * c) {
  int nthreads = gridDim.x * blockDim.x;
  int tid = blockDim.x*blockIdx.x + threadIdx.x;

  for (int i=tid; i<n; i+=nthreads)
    c[i] = a[i] + b[i];
}

#ifdef DEBUG
void print_vec(int n, double * v) {
  for (int i=0; i<n; i++)
    printf("%5.2lf ", v[i]);
  printf("\n");
}
#endif

int main(int argc, char * argv[]) {
  int N, err;

  assert(argc == 2);
  N = atoi(argv[1]);
  assert(N>=1);

  printf("N = %d\n", N);
  fflush(stdout);
  double * a = (double*)malloc(N*sizeof(double));
  assert(a);
  double * b = (double*)malloc(N*sizeof(double));
  assert(b);
  double * c = (double*)malloc(N*sizeof(double));
  assert(c);
  for (int i=0; i<N; i++) {
    a[i] = sin(i);
    b[i] = cos(i);
  }
  printf("Host initialization complete.\n");
  fflush(stdout);
#ifdef DEBUG
  print_vec(N, a);
  print_vec(N, b);
#endif
  
  double * a_dev, * b_dev, * c_dev, start_time = mytime();

  err = cudaMalloc((void**)&a_dev, N*sizeof(double));
  assert(err == cudaSuccess);
  err = cudaMalloc((void**)&b_dev, N*sizeof(double));
  assert(err == cudaSuccess);
  err = cudaMalloc((void**)&c_dev, N*sizeof(double));
  assert(err == cudaSuccess);
  err = cudaMemcpy(a_dev, a, N*sizeof(double), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  err = cudaMemcpy(b_dev, b, N*sizeof(double), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  printf("Device initialization complete.\n");
  fflush(stdout);
  vec_add<<<nblocks, threadsPerBlock>>>(N, a_dev, b_dev, c_dev);
  cudaMemcpy(c, c_dev, N*sizeof(double), cudaMemcpyDeviceToHost);
  printf("Result obtained.  Time: %lf\n", mytime() - start_time);
#ifdef DEBUG
  print_vec(N, c);
#endif
  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  free(a);
  free(b);
  free(c);
}
