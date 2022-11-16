#include <stdio.h>
#include <assert.h>
#define IN(i,j) in_dev[(i)*m+(j)]
#define OUT(i,j) out_dev[(i)*m2+(j)]

const int nthreadsx = 32, nthreadsy = 32;

/* Assigns one pixel in out_dev the average of 4 corresponding pixels
 * in in_dev.  n2=n/2 and m2=m/2 are the dimensions of out_dev.
 */
__global__ void shrink(int n2, int m2, float * in_dev, float * out_dev) {
  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  const int j = blockDim.y*blockIdx.y + threadIdx.y;
  const int m = 2*m2;

  if (i < n2 && j < m2)
    OUT(i,j) = (IN(2*i, 2*j) + IN(2*i+1, 2*j) +
		IN(2*i, 2*j+1) + IN(2*i+1, 2*j+1))/4;
}

void print2d(int n, int m, float * a) {
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++)
      printf("%5.2f ", a[i*m+j]);
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  float * in, * out, * in_dev, * out_dev;

  const int n = atoi(argv[1]);
  const int m = atoi(argv[2]);
  assert (n%2==0);
  assert (m%2==0);
  const int n2 = n/2;
  const int m2 = m/2;
  in = (float*)malloc(n*m*sizeof(float));
  out = (float*)malloc(n2*m2*sizeof(float));
  assert(in); assert(out);
  cudaMalloc((void**)&in_dev, n*m*sizeof(float));
  cudaMalloc((void**)&out_dev, n2*m2*sizeof(float));
  assert(in_dev); assert(out_dev);
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      in[i*m+j] = (float)(i*m+j);
  printf("Input:\n");
  print2d(n, m, in);
  printf("\nOutput:\n");
  cudaMemcpy(in_dev, in, n*m*sizeof(float), cudaMemcpyHostToDevice);
  const int nblocksx = n2/nthreadsx + (n2%nthreadsx != 0);
  const int nblocksy = m2/nthreadsy + (m2%nthreadsy != 0);
  const dim3 blockDim(nthreadsx, nthreadsy), gridDim(nblocksx, nblocksy);
  shrink<<<gridDim, blockDim>>>(n2, m2, in_dev, out_dev);
  cudaMemcpy(out, out_dev, n2*m2*sizeof(float), cudaMemcpyDeviceToHost);
  print2d(n2, m2, out);
}

 