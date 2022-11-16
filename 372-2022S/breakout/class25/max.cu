#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#define MIN(a,b) ((a)<(b)?(a):(b))
#define T double

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

/* Does most of the work in computing the max of a.  a is an array of
   length N.  c is an array of length nblocks.  Upon return c[i] will
   hold max of the portion of the array corresponding to the indexes
   for which block i is responsible.  The final max is the max over
   all blocks i of c[i]. */
__global__ void max(int n, T * a, T * c) {
  __shared__ T maxs[threadsPerBlock];
  const int ltid = threadIdx.x; // local thread ID (within this block)
  const int gtid = ltid + blockIdx.x * blockDim.x; // global thread ID
  const int nthreads = gridDim.x * blockDim.x;
  T thread_max = gtid < n ? a[gtid] : 0;

  for (int i = gtid + nthreads; i < n; i += nthreads) {
    if (a[i] > thread_max) thread_max = a[i];
    //thread_max = fmax(thread_max, a[i]);
  }
  maxs[ltid] = thread_max;
  __syncthreads();  // barrier for the threads in this block
  // reduction over the block. threadsPerBlock must be a power of 2...
  for (int i = blockDim.x/2; i > 0; i /= 2) {
    if (ltid < i) {
      // maxs[ltid] = fmax(maxs[ltid], maxs[ltid + i]);
      if (maxs[ltid + i] > maxs[ltid])
      	maxs[ltid] = maxs[ltid + i];
    }
    __syncthreads();
  }
  // at this point, maxs[0] holds the max over all threads.
  if (ltid == 0) c[blockIdx.x] = maxs[0];
}

int main() {
  T * a, * partial_maxs, * dev_a, * dev_partial_maxs;
  int err;
  double start_time = mytime();
  
  printf("max: N = %d, threadsPerBlock = %d, nblocks = %d, nthreads = %d\n",
	 N, threadsPerBlock, nblocks, threadsPerBlock*nblocks);
  a = (T*)malloc(N*sizeof(T));
  assert(a);
  partial_maxs = (T*)malloc(nblocks*sizeof(T));
  err = cudaMalloc(&dev_a, N*sizeof(T));
  assert(err == cudaSuccess);
  err = cudaMalloc(&dev_partial_maxs, nblocks*sizeof(T));
  for (int i = 0; i < N; i++) {
    a[i] = sin(i);
  }
  err = cudaMemcpy(dev_a, a, N*sizeof(T), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  max<<<nblocks, threadsPerBlock>>>(N, dev_a, dev_partial_maxs);
  err = cudaMemcpy(partial_maxs, dev_partial_maxs, nblocks*sizeof(T),
		   cudaMemcpyDeviceToHost);
  assert(err == cudaSuccess);
  cudaFree(dev_a);
  cudaFree(dev_partial_maxs);

  T result = partial_maxs[0];

  for (int i = 1; i < nblocks; i++) {
    if (partial_maxs[i] > result) result = partial_maxs[i];
  }
  printf("Result = %.20lf.  Time = %lf\n", result, mytime() - start_time);
  free(a);
  free(partial_maxs);
}
