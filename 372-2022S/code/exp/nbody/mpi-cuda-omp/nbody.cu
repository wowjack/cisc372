/* nbody.cu: MPI/CUDA/OpenMP 2-d nbody simulation
   Author: Stephen Siegel

   Link this with a translation unit that defines the extern
   variables, and anim.o, to make a complete program.
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "nbody.h"
#include "cudaanim.h"

/* Global variables */
extern const double x_min;     /* coord of left edge of window (meters) */
extern const double x_max;     /* coord of right edge of window (meters) */
extern const double y_min;     /* coord of bottom edge of window (meters) */
extern const double y_max;     /* coord of top edge of window (meters) */
extern const int nx;           /* width of movie window (pixels) */
extern const int nbodies;      /* number of bodies */
extern const double delta_t;   /* time between discrete time steps (seconds) */
extern const int nstep;        /* number of time steps */
extern const int wstep;        /* number of times steps beween movie frames */
extern const int ncolors;      /* number of colors to use for the bodies */
extern const int colors[][3];  /* colors we will use for the bodies */
extern const Body bodies[];    /* list of bodies with initial data */
const double G = 6.674e-11;    /* universal gravitational constant */
int ny;                        /* height of movie window (pixels) */
State * states, * states_new;  /* two copies of state array */
ANIM_File af;                  /* output anim file */
double * posbuf;               /* to send data to anim, 2*nbodies doubles */
double start_time;             /* time simulation starts */
int nprocs;
int rank;

__global__ void kernel(int rank) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  printf("Hello from block %d, thread %d of the GPU, called by process %d\n",
	 bid, tid, rank);
}

/* One argument: the name of the output file */
int main(int argc, char * argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int statbar = 0; // used for printing status updates

    assert(argc == 2);
    printf("nbodies=%d nstep=%d wstep=%d\n", nbodies, nstep, wstep);
    printf("G = %E\n", G);
    for (int i=1; i<=nstep; i++) {
      ANIM_Status_update(stdout, nstep, i, &statbar);
    }
    printf("\n");
  }
  kernel<<<3,4>>>(rank);
#pragma omp parallel for
  for (int i=0; i<8; i++) {
    printf("Proc %d: OpenMP Loop Iteration %d\n", rank, i);
  }
  printf("\n");
  cudaDeviceSynchronize();
  MPI_Finalize();
}
