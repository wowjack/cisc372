#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

const int N = 1u<<30;
const int threadsPerBlock = 256;
// use at most 120 blocks.  k40c has 15 SMPs, so that's 8 blocks per
// SMP.  For small values of N, we will use less than 120
// blocks...just enough to have one index per thread...
const int nblocks = MIN(120, (N + threadsPerBlock - 1) / threadsPerBlock);

double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

/* Does most of the work in computing the dot product of a and b.  a
   and b are arrays of length N.  c is an array of length nblocks.
   Upon return c[i] will hold the portion of the dot product
   corresponding to the indexes for which block i is responsible.  The
   final dot product is the sum over all blocks i of c[i]. */
__global__ void dot(float * a, float * b, float * c) {
  __shared__ float sums[threadsPerBlock];
  const int ltid = threadIdx.x; // local thread ID (within this block)
  const int gtid = ltid + blockIdx.x * blockDim.x; // global thread ID
  const int nthreads = gridDim.x * blockDim.x;
  float thread_sum = 0;

  for (int i = gtid; i < N; i += nthreads) thread_sum += a[i] * b[i];
  sums[ltid] = thread_sum;
  __syncthreads();  // barrier for the threads in this block
  // reduction over the block. threadsPerBlock must be a power of 2...
  for (int i = blockDim.x/2; i > 0; i /= 2) {
    if (ltid < i) sums[ltid] += sums[ltid + i];
    __syncthreads();
  }
  // at this point, sums[0] holds the sum over all threads.
  if (ltid == 0) c[blockIdx.x] = sums[0];
}

int main() {
  float * a, * b, * partial_sums, * dev_a, * dev_b, * dev_partial_sums;
  int err;
  double start_time = mytime();
  
  printf("dot: N = %d, threadsPerBlock = %d, nblocks = %d, nthreads = %d\n",
	 N, threadsPerBlock, nblocks, threadsPerBlock*nblocks);
  a = (float*)malloc(N*sizeof(float));
  assert(a);
  b = (float*)malloc(N*sizeof(float));
  assert(b);
  partial_sums = (float*)malloc(nblocks*sizeof(float));
  err = cudaMalloc((void**)&dev_a, N*sizeof(float));
  assert(err == cudaSuccess);
  err = cudaMalloc((void**)&dev_b, N*sizeof(float));
  assert(err == cudaSuccess);
  err = cudaMalloc((void**)&dev_partial_sums, nblocks*sizeof(float));
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i*2;
  }
  err = cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  err = cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice); 
  assert(err == cudaSuccess);
  dot<<<nblocks, threadsPerBlock>>>(dev_a, dev_b, dev_partial_sums);
  err = cudaMemcpy(partial_sums, dev_partial_sums, nblocks*sizeof(float),
		   cudaMemcpyDeviceToHost);
  assert(err == cudaSuccess);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_partial_sums);

  float result = 0.0f;
  float expected = 2 * sum_squares((float)(N - 1));

  for (int i = 0; i < nblocks; i++) result += partial_sums[i];
  printf("Result = %.12g.  Expected = %.12g.  Time = %lf\n",
	 result, expected, mytime() - start_time);
  fflush(stdout);
  assert(result/expected <= 1.0001);
  assert(expected/result <= 1.0001);
  free(a);
  free(b);
  free(partial_sums);
}

