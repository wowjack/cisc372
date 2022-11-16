#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

const int nthreadsx=8, nthreadsy=8;

double* matrix;
double* matrix_dev;
double* result;
double* result_dev;

double* cpuAlloc(int n, int m){
    double* matrix = (double*)malloc(n*m*sizeof(double));
    return matrix;
}

double* gpuAlloc(int n, int m){
    double* matrix;
    cudaMalloc(&matrix, n*m*sizeof(double));
    return matrix;
}

__global__ void shrink(int n, int m){
    int threadx = blockIdx.x * blockDim.x + threadIdx.x;
    int thready = blockIdx.y * blockDim.y + threadIdx.y;
    //map 2d thread coordinates to the 1d array we are treating as 2d
    int idx = threadx*n + thready;

    
}

int main(int argc, char* argv[]){
    assert(argc == 3);
    int n = atoi(argv[1]), m = atoi(argv[2]);
    assert(n%2 == 0); assert(m%2 == 0);

    //Allocate the big matrices for device and host
    matrix = cpuAlloc(n, m);
    matrix_dev = gpuAlloc(n, m);

    //Allocate the small matrices for device and host
    result = cpuAlloc(n/2, m/2);
    result_dev = gpuAlloc(n/2, m/2);

    //Get the grid and block dimensions
    //Note we only need one thread for each result matrix cell, this is 4 times too many
    dim3 gridDim(n/nthreadsx + (n % nthreadsx != 0), m/nthreadsy + (n % nthreadsy != 0));
    dim3 blockDim(nthreadsx, nthreadsy);

    //Send the host matrix to the device matrix
    cudaMemcpy(matrix_dev, matrix, n*m, cudaMemcpyHostToDevice);
    //Do the shrinking
    shrink<<<gridDim, blockDim>>>(n, m);
    //Copy the result matrix from device to host
    cudaMemcpy(result, result_dev, n/2*m/2, cudaMemcpyDeviceToHost);

    return 0;
}