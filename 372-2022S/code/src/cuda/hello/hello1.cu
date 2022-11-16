#include <stdio.h>

__global__ void kernel(void) {
  printf("Hello from the GPU!\n");
}

int main (void) {
  kernel<<<1,1>>>();
  printf("Hello from the CPU!\n");
  cudaDeviceSynchronize();
}
