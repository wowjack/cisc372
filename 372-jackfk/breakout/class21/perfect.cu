/* perfect.c: find all perfect numbers up to a bound */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

__device__ int is_perfect(int n) {
  if (n < 2) return 0;
  int sum = 1, i = 2;
  while (1) {
    const int i_squared = i*i;
    if (i_squared < n) {
      if (n%i == 0) sum += i + n/i;
      i++;
    } else {
      if (i_squared == n) sum += i;
      break;
    }
  }
  return sum == n;
}

__global__ void get_perfects(int bound){
  int totalThreads = blockDim.x * gridDim.x;
  int globalThread = blockDim.x * blockIdx.x + threadIdx.x;
  //Going to use a cyclic distribution
  for(int i=globalThread; i<=bound; i+=totalThreads){
    if (i%1000000 == 0) {
      //printf("i = %d\n", i);
    }
    if (is_perfect(i)) {
      printf("Found a perfect number: %d\n", i);
    }
  }
}

int main(int argc, char * argv[]) {
  double start_time = mytime();

  //Explain usage
  if (argc != 2) {
    printf("Usage: perfect.exec bound\n");
    exit(1);
  }

  int threadsPerBlock = 1024, blocks = 15;
  printf("%d blocks with %d threads per block making %d total threads.\n", blocks, threadsPerBlock, blocks*threadsPerBlock);

  get_perfects<<<blocks,threadsPerBlock>>>(atoi(argv[1]));
  cudaDeviceSynchronize();

  printf("Time = %lf\n", mytime() - start_time);
}
