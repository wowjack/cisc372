/* diffusion2d.cu: CUDA version of diffusion2d.c */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "cudaanim.h"

/* Constants */
const int nthreadsx = 8, nthreadsy = 8;

/* Global variables */
int nx, ny;               /* dimensions of the room (in pixels) */
int width;                /* ny*sizeof(double) */
double k;                 /* constant controlling rate of diffusion */
int nstep;                /* number of time steps */
int wstep;                /* time between writes to file */
char * filename;          /* name of file to create */
double * u;               /* temperature data on CPU (nn*ny doubles) */
size_t pitch;             /* the CUDA "pitch" used for 2d array u_dev */
double * u_dev;           /* temperature function in GPU global memory */
double * u_new_dev;       /* second copy of temperature on GPU */
ANIM_File af;             /* output file */
double start_time;        /* time simulation starts */
int nblocksx, nblocksy;   /* number of blocks in x and y directions */

static void quit() {
  printf("Usage: diffusion2d.exec NX NY K NSTEPS WSTEP FILENAME         \n\
  NX = number of pixels in x-direction                                  \n\
  NY = number of pixels in y-direction                                  \n\
  K =  a constant controlling rate of diffusion in (0,.25)              \n\
  NSTEP = total number of time steps, at least 1                        \n\
  WSTEP = number of time steps between writes to file, in [0, NSTEP]    \n\
  FILENAME = name of output file                                        \n\
Example: diffusion2d.exec 400 400 0.2 50000 100 out.anim                \n");
  exit(1);
}

static void setup(int argc, char * argv[]) {
  if (argc != 7) quit();
  start_time = ANIM_time();
  nx = atoi(argv[1]), ny = atoi(argv[2]), k = atof(argv[3]),
    nstep = atoi(argv[4]), wstep = atoi(argv[5]), filename = argv[6];
  if (!(nx>=6 && ny>=6 && 0<k && k<.25 && nstep>=1 && wstep>=0 && wstep<=nstep))
    quit();
  width = ny*sizeof(double);
  nblocksx = (nx-2)/nthreadsx + ((nx-2)%nthreadsx != 0);
  nblocksy = (ny-2)/nthreadsy + ((ny-2)%nthreadsy != 0);

  const int nframes = wstep == 0 ? 0 : 1+nstep/wstep;
  int err, dims[] = {nx, ny};
  ANIM_range_t ranges[] = {{0., 1.*nx}, {0., 1.*ny}, {0., 100.}};
  size_t pitch2;
  
  u = (double*)malloc(nx*width);
  assert(u);
  err = cudaMallocPitch(&u_dev, &pitch, width, nx);
  assert(err == cudaSuccess);
  cudaMallocPitch(&u_new_dev, &pitch2, width, nx);
  assert(err == cudaSuccess);
  assert(pitch == pitch2);
  printf("diffusion2d: nx=%d ny=%d k=%.3lf nstep=%d wstep=%d pitch=%zu nblocks=(%d,%d)\n",
	 nx, ny, k, nstep, wstep, pitch, nblocksx, nblocksy);
  printf("diffusion2d: creating ANIM file %s with %d frames, %zu bytes.\n",
	 filename, nframes, ANIM_Heat_file_size(2, dims, nframes));
  fflush(stdout);
  for (int i = 1; i < nx - 1; i++)
    for (int j = 1; j < ny - 1; j++)
      u[i*ny+j] = 100.0;
  for (int i=0; i < nx; i++)
    u[i*ny+0] = u[i*ny+ny-1] = 0.0;
  for (int j=1; j < ny - 1; j++)
    u[0*ny+j] = u[(nx-1)*ny+j] = 0.0;
  err = cudaMemcpy2D(u_dev, pitch, u, width, width, nx,
		     cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  err = cudaMemcpy2D(u_new_dev, pitch, u, width, width, nx,
		     cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);
  af = ANIM_Create_heat(2, dims, ranges, filename);
}

static void teardown() {
  ANIM_Close(af);
  free(u);
  cudaFree(u_dev);
  cudaFree(u_new_dev);
  printf("\ndiffusion2d: finished.  Time = %lf\n", ANIM_time() - start_time);
}

#define get(t,i,j) ((double*)((char*)(t) + (i)*(pitch)))[j]

__global__ void update(int nx, int ny, int pitch,
		       double k, double * t, double * t_new) {
  const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;
  const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;

  if (i < nx-1 && j < ny-1) {
    const double tij = get(t,i,j);
    get(t_new,i,j) = tij +
      k*(get(t,i+1,j) + get(t,i-1,j) + get(t,i,j-1) + get(t,i,j+1) - 4*tij);
  }
}

int main(int argc, char *argv[]) {
  setup(argc, argv);

  int err, dots = 0; // number of dots printed so far (0..100)
  const dim3 blockDim(nthreadsx, nthreadsy), gridDim(nblocksx, nblocksy);
  
  if (wstep != 0) ANIM_Write_frame(af, u);
  for (int i = 1; i <= nstep; i++) {
    update<<<gridDim,blockDim>>>(nx, ny, pitch, k, u_dev, u_new_dev);
    double * const tmp = u_new_dev; u_new_dev = u_dev; u_dev = tmp;
    ANIM_Status_update(stdout, nstep, i, &dots);
    if (wstep != 0 && i%wstep == 0) {
      err = cudaMemcpy2D(u, width, u_dev, pitch, width, nx,
			 cudaMemcpyDeviceToHost);
      assert(err == cudaSuccess);
      ANIM_Write_frame(af, u);
    }
  }
  teardown();
}
