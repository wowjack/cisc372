#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define N pow(2, 10)
#define ThreadsPerBlock 1024
#define Blocks 32

double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

double* gpuAlloc(){
    double* buf;
    cudaMalloc(&buf, Blocks*sizeof(double));
    return buf;
}
double* cpuAlloc(){
    return (double*)malloc(Blocks*sizeof(double));
}

__global__ void getMaxs(double* buf, int stopNum){
    int totalThreads = gridDim.x * blockDim.x;
    int threadNum = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = threadNum; i < stopNum; i += totalThreads){
        if(sinf(i) > buf[i]) buf[i] = sinf(i);
    }
}

double getMaxFromArr(double* arr, int size){
    double max = 0;
    for(int i=0; i<size; i++){
        if(arr[i] > max) max = arr[i];
    }
    return max;
}

int main(){
    double* maxs = cpuAlloc();
    double* maxs_device = gpuAlloc();

    double startTime = mytime();
    
    getMaxs<<<Blocks, ThreadsPerBlock>>>(maxs_device, (int)N);
    cudaMemcpy(maxs, maxs_device, Blocks, cudaMemcpyDeviceToHost);
    double max = getMaxFromArr(maxs, Blocks);
    printf("Maximum: %f\n", max);
    printf("Runtime: %f\n", mytime() - startTime);
    return EXIT_SUCCESS;
}