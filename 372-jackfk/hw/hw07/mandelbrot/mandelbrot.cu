/*  Based on fractal code by Martin Burtscher. */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "cudaanim.h"

const double Delta = 0.001;
const double xMid =  0.23701;
const double yMid =  0.521;

double* buf; double* buf_dev;

const int threadsPerBlock = 32; // 32x32 block dim

static void quit() {
  printf("Usage: mandelbrot.exec WIDTH NSTEP FILENAME            \n\
  WIDTH = frame width, in pixels, (at least 10)                  \n\
  NSTEP = number of frames in the animation (at least 1)         \n\
  FILENAME = name of output file (to be created)                 \n\
Example: mandelbrot.exec 200 100 out.anim                        \n");
  exit(1);
}

//Kernel that computes the depth for each cell in the matrix
__global__ void  update(double delta, double xMin, double yMin, int width, double* buf_dev) {
    //I tried so goddamn hard to use shared memory to make it faster but I just
    //couldn't get it right for the life of me.
    //I've got finals to take and other homework to do so I give up.
    const double dw = 2.0 * delta / width;

    const int threadx = blockDim.x * blockIdx.x + threadIdx.x,
              thready = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(threadx < width && thready < width) {
        const double cx = xMin + thready * dw;
        const double cy = yMin + threadx * dw;
        double x = cx, y = cy, x2, y2;
        int depth = 256;
        do {
            x2 = x * x;
            y2 = y * y;
            y = 2 * x * y + cy;
            x = x2 - y2 + cx;
            depth--;
        } while (depth > 0 && x2 + y2 < 5.0);
        buf_dev[thready*width+threadx] = (double)depth;
    }
}


void print(double* buf, int dim) {
    for(int i = 0; i<dim; i++){
        for(int j = 0; j<dim; j++){
            printf("%.1f ", buf[i*dim+j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc != 4) quit();
    double start_time = ANIM_time();
    int dots = 0, width = atoi(argv[1]), nstep = atoi(argv[2]);
    int dims[] = {width, width};
    char * filename = argv[3];

    if (nstep < 1) quit();

    printf("mandelbrot: creating ANIM file %s with %d frames, %dx%d pixels, %zu bytes.\n",
	    filename, nstep, width, width,
	    ANIM_Heat_file_size(2, dims, nstep));

    ANIM_range_t ranges[] = {{0, (double)width}, {0, (double)width}, {0, 255}};
    ANIM_File af =
    ANIM_Create_heat(2, dims, ranges, filename);

    //Allocate the matrices on host and device
    double * buf = (double*)malloc(width * width * sizeof(double)), delta = Delta;
    cudaMalloc(&buf_dev, width*width*sizeof(double));

    assert(buf); assert(buf_dev);

    const dim3 blockDim(threadsPerBlock, threadsPerBlock),
               gridDim(width/threadsPerBlock + (width%threadsPerBlock!=0), width/threadsPerBlock + (width%threadsPerBlock!=0));

    for (int frame = 0; frame < nstep; frame++) {
        const double xMin = xMid - delta, yMin = yMid - delta;
        update<<<gridDim, blockDim>>>(delta, xMin, yMin, width, buf_dev);

        //copy the device matrix back to host
        cudaMemcpy(buf, buf_dev, width*width*sizeof(double), cudaMemcpyDeviceToHost);

        ANIM_Write_frame(af, buf);
        ANIM_Status_update(stdout, nstep, frame+1, &dots);

        delta *= 0.99;
    }
    ANIM_Close(af);
    printf("\nmandelbrot: finished.  Time = %lf\n", ANIM_time() - start_time);
    free(buf);
    cudaFree(buf_dev);
}
