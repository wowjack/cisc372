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

/* Number of threads per block in x and y direction. 32*32=1024 */
const int nthreadsx = 32, nthreadsy = 32;

double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

__global__ void mat_add(int n, int m, double * a, double * b, double * c) {
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x < n && y < m) {
    const int idx = x*m + y;
    c[idx] = a[idx] + b[idx];
  }
}

#ifdef DEBUG
void print_mat(int n, int m, double * a) {
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++)
      printf("%5.2lf ", a[i*m + j]);
    printf("\n");
  }
}
#endif

int main(int argc, char * argv[]) {
  assert(argc == 3);

  int err, n = atoi(argv[1]), m = atoi(argv[2]);
  
  assert(n>=1 && m>=1);

  const int nblocksx = n/nthreadsx + (n%nthreadsx != 0);
  const int nblocksy = m/nthreadsy + (m%nthreadsy != 0);
  const dim3 blockDim(nthreadsx, nthreadsy), gridDim(nblocksx, nblocksy);
  
  printf("size=(%d, %d), nblocks=(%d, %d), nthreads=(%d, %d)\n",
	 n, m, nblocksx, nblocksy, nthreadsx, nthreadsy);
  fflush(stdout);
  
  double * a = (double*)malloc(n*m*sizeof(double));  assert(a);
  double * b = (double*)malloc(n*m*sizeof(double));  assert(b);
  double * c = (double*)malloc(n*m*sizeof(double));  assert(c);

  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      a[i*m+j] = sin(i*m+j);
      b[i*m+j] = cos(i*m+j);
    }
  }
  printf("Host initialization complete.\n");  fflush(stdout);
#ifdef DEBUG
  print_mat(n, m, a);
  printf("\n");
  print_mat(n, m, b);
#endif
  
  double * a_dev, * b_dev, * c_dev, start_time = mytime();

  err = cudaMalloc((void**)&a_dev, n*m*sizeof(double));
  assert(err == cudaSuccess);
  err = cudaMalloc((void**)&b_dev, n*m*sizeof(double));
  assert(err == cudaSuccess);
  err = cudaMalloc((void**)&c_dev, n*m*sizeof(double));
  assert(err == cudaSuccess);
  err = cudaMemcpy(a_dev, a, n*m*sizeof(double), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  err = cudaMemcpy(b_dev, b, n*m*sizeof(double), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  printf("Device initialization complete.\n");  fflush(stdout);
  mat_add<<<gridDim, blockDim>>>(n, m, a_dev, b_dev, c_dev);
  cudaMemcpy(c, c_dev, n*m*sizeof(double), cudaMemcpyDeviceToHost);
  printf("Result obtained.  Time: %lf\n", mytime() - start_time);
#ifdef DEBUG
  print_mat(n, m, c);
#endif
  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  free(a);
  free(b);
  free(c);
}
