/* Simplest CUDA version of diffuse1d.  Each thread operates on one
   element, no shared memory. */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

#define threadsPerBlock 1024

/* Global variables */
const double m = 100.0;   /* initial temperature of rod interior */
int nx;                   /* number of discrete points including endpoints */
double k;                 /* D*dt/(dx*dx) */
int nstep;                /* number of time steps */
int wstep;                /* time between writes to file */
double * u;               /* host copy of temperature array */
double * u_dev;           /* device copy of temperature array */
double * u_new_dev;       /* second copy of temp array on device */
double start_time;        /* time simulation starts */
FILE * out;               /* where the (plain text) output goes */

static void quit() {
  printf("Usage: diffuse1d.exec NX K NSTEPS WSTEP [FILENAME]          \n\
  NX = number of points in rod, including the two endpoints           \n\
  K = D*dt/(dx*dx), a constant conrolling rate of diffusion in (0,.5) \n\
  NSTEPS = total number of time steps                                 \n\
  WSTEP = number of time steps between writes to file                 \n\
  FILENAME = file to send output to (optional)                        \n\
Example: diffuse1d.exec 100 0.3 1000 10                               \n");
  exit(1);
}

static double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

static void setup(int argc, char * argv[]) {
  if (argc < 5 ||argc > 6) quit();
  nx = atoi(argv[1]), k = atof(argv[2]), nstep = atoi(argv[3]),
    wstep = atoi(argv[4]);
  out = argc == 5 ? stdout : fopen(argv[5], "w");
  assert(out);
  if (!(nx>=2 && 0<k && k<.5 && nstep>=1 && wstep>=0 && wstep<=nstep))
    quit();
  printf("Starting diffuse1d: nx=%d k=%lf nstep=%d wstep=%d\n",
	 nx, k, nstep, wstep);
  fflush(stdout);
  start_time = mytime();
  u = (double*)malloc(nx*sizeof(double));
  assert(u);
  int err = cudaMalloc((void**)&u_dev, nx*sizeof(double));
  assert(err == cudaSuccess);
  err = cudaMalloc((void**)&u_new_dev, nx*sizeof(double));
  assert(err == cudaSuccess);
  for (int i = 1; i < nx - 1; i++) u[i] = m;
  u[0] = u[nx-1] = 0.0;
  cudaMemcpy(u_dev, u, nx*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(u_new_dev, u, nx*sizeof(double), cudaMemcpyHostToDevice);
}

static void teardown() {
  free(u);
  cudaFree(u_dev);
  cudaFree(u_new_dev);
  if (out != stdout) fclose(out);
  printf("Finished diffuse1d: time = %lf\n", mytime() - start_time);
}

static void write(int time) {
  fprintf(out, "%4d: ", time);
  for (int i = 0; i < nx; i++)
    fprintf(out, "%7.2lf ", u[i]);
  fprintf(out, "\n");
  fflush(out);
}

__global__ static void
update(int n, double k, double * t_old, double * t_new) {
  const int idx = blockIdx.x*blockDim.x + threadIdx.x + 1;

  if (0 < idx && idx < n-1) {
    const double tmp = t_old[idx];
    t_new[idx] = tmp + k*(t_old[idx+1] + t_old[idx-1] - 2*tmp);
  }
}

int main(int argc, char *argv[]) {
  setup(argc, argv);
  const int nblocks = (nx-2)/threadsPerBlock + (0 != (nx-2)%threadsPerBlock);
  if (wstep != 0) write(0);
  for (int i = 1; i <= nstep; i++) {
    update<<<nblocks,threadsPerBlock>>>(nx, k, u_dev, u_new_dev);
    double * const tmp = u_new_dev; u_new_dev = u_dev; u_dev = tmp;
    if (wstep != 0 && i%wstep == 0) {
      cudaMemcpy(u, u_dev, nx*sizeof(double), cudaMemcpyDeviceToHost);
      write(i);
    }
  }
  teardown();
}
