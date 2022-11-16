/* In this CUDA version of diffuse1d, each thread operates on a chunk
   of elements, and the elements are all loaded into shared memory
   first. */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#define MAX(p,q) ((p)>(q) ? (p) : (q))
#define MIN(p,q) ((p)<(q) ? (p) : (q))

#ifndef threadsPerBlock
#define threadsPerBlock 256
#endif
// chunkSize is the number of cells each thread will handle
#ifndef chunkSize
#define chunkSize 16
#endif
#define blockSize (threadsPerBlock*chunkSize)

/* Global variables */
const double m = 100.0;   /* initial temperature of rod interior */
int nx;                   /* number of discrete points including endpoints */
double k;                 /* D*dt/(dx*dx) */
int nstep;                /* number of time steps */
int wstep;                /* time between writes to file */
double * u;
double * u_dev, * u_new_dev;
double start_time;        /* time simulation starts */
FILE * out;               /* where the output goes */

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

// Global indexes are in 0..n-1.    Indexes 0 and n-1 are fixed boundaries.
// Global indexes 1..n-2 are block distributed.

__global__ static void update(int n, double k, double * t_old, double * t_new) {
  __shared__ double tmp[blockSize+2];
  // global index of first element of this block (local index 1)...
  const int blockFirst = blockIdx.x*blockSize + 1;
  // global index of the last entry of this block...
  const int blockLast = MIN(n-2, (blockIdx.x+1)*blockSize);
  // local index of the last entry of this block...
  // const int blockLast_local = blockLast - blockFirst + 1;
  // global index of first element owned by this thread...
  const int threadFirst = blockFirst + threadIdx.x*chunkSize;
  // first index of tmp this thread is updating...
  const int threadFirst_local = threadFirst - blockFirst + 1;
  // number of cells owned by this thread...
  const int numOwned = MIN(chunkSize, n-1-threadFirst);
  // am I the last thread of this block (owner of blockLast)?...
  const int amLast =
    (threadFirst <= blockLast && blockLast < threadFirst + numOwned);
  // adjustment for thread 0 to load left ghost cell...
  const int copyStart = threadIdx.x == 0 && numOwned > 0 ? -1 : 0;
  // adjustment for last thread to load right ghost cell...
  const int copyStop = amLast ? numOwned + 1 : numOwned;
#ifdef DEBUG
  if (numOwned > 0)
    printf("Block %d Thread %d: blockLast=%d, amLast=%d copyStart=%d copyStop=%d\n",
  	   blockIdx.x, threadIdx.x, blockLast, amLast, copyStart, copyStop);
#endif
  // load the chunk for this thread into shared memory...
  for (int i=copyStart; i<copyStop; i++)
    tmp[threadFirst_local + i] = t_old[threadFirst + i];
 
  __syncthreads();  // barrier for this block

  for (int i=0; i<numOwned; i++)
    t_new[threadFirst + i] = tmp[threadFirst_local + i]
      + k*(  tmp[threadFirst_local + i + 1]
	   + tmp[threadFirst_local + i - 1]
	   - 2*tmp[threadFirst_local + i] );
}

int main(int argc, char *argv[]) {
  setup(argc, argv);
  const int nblocks = (nx-2)/blockSize + (0 != (nx-2)%blockSize);
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
