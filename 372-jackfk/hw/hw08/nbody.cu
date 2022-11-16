
//I am using omp and cuda
#include <omp.h>

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "nbody.h"
#include "cudaanim.h"

#define ThreadsPerBlock 1024

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
Body* bodies_dev;
const double G = 6.674e-11;    /* universal gravitational constant */
int ny;                        /* height of movie window (pixels) */
State * states;                /* The state array */
State* states_dev, * states_dev_new; /* State arrays for the gpu to read and write to */
ANIM_File af;                  /* output anim file */
double * posbuf;               /* to send data to anim, 2*nbodies doubles */
double start_time;             /* time simulation starts */

static void init(char* filename) {
  start_time = ANIM_time();
  assert(x_max > x_min && y_max > y_min);
  ny = ceil(nx*(y_max - y_min)/(x_max - x_min));
  printf("nbody: nbodies=%d nx=%d ny=%d nstep=%d wstep=%d\n",
	 nbodies, nx, ny, nstep, wstep);
  const int nframes =  wstep == 0 ? 0 : 1+nstep/wstep;
  printf("nbody: creating ANIM file %s with %d frames, %zu bytes.\n",
	 filename, nframes,
	 ANIM_Nbody_file_size(2, nbodies, ncolors, nframes));
  fflush(stdout);
  assert(nx >= 10 && ny >= 10);
  assert(nstep >= 1 && wstep >= 0 && nbodies > 0);
  assert(ncolors >= 1 && ncolors <= ANIM_MAXCOLORS);
  states = (State*)malloc(nbodies * sizeof(State));
  cudaMalloc(&states_dev, nbodies * sizeof(State)); //Allocate the states array on the gpu
  assert(states);
  cudaMalloc(&states_dev_new, nbodies * sizeof(State)); //Allocate the states_new array on the gpu
  posbuf = (double*)malloc(2 * nbodies * sizeof(double));
  assert(posbuf);

  int radii[nbodies], bcolors[nbodies];
  ANIM_color_t acolors[ncolors]; // RGB colors converted to ANIM colors

  #pragma omp parallel
  {
  #pragma omp for nowait
  for (int i=0; i<nbodies; i++) {
    assert(bodies[i].mass > 0);
    assert(bodies[i].color >= 0 && bodies[i].color < ncolors);
    assert(bodies[i].radius > 0);
    states[i] = bodies[i].state;
    radii[i] = bodies[i].radius;
    bcolors[i] = bodies[i].color;
  }
  #pragma omp for
  for (int i=0; i<ncolors; i++)
    acolors[i] = ANIM_Make_color(colors[i][0], colors[i][1], colors[i][2]);
  }

  //copy the initial bodies array to the gpu, this only needs to be done once
  cudaMalloc(&bodies_dev, nbodies*sizeof(Body));
  cudaMemcpy(bodies_dev, bodies, nbodies*sizeof(Body), cudaMemcpyHostToDevice);

  //also copy the states array to the gpu, this only needs to be done once
  cudaMemcpy(states_dev, states, nbodies*sizeof(State), cudaMemcpyHostToDevice);
  
  int size[] = {nx, ny};
  ANIM_range_t ranges[] = {{y_min, y_max}, {x_min, x_max}};
  af = ANIM_Create_nbody(2, size, ranges,
			 nbodies, radii, ncolors, acolors, bcolors, filename);
}

static inline void write_frame() {
  for (int i=0, j=0; i<nbodies; i++) {
    posbuf[j++] = states[i].x;
    posbuf[j++] = states[i].y;
  }
  ANIM_Write_frame(af, posbuf);
}


__global__ void update(State* states, State* states_new, Body* bodies, int nbodies, double delta_t) {
  //No fancy 2d grid or 2d blocks are needed
  const int thread = blockDim.x * blockIdx.x + threadIdx.x;

  if(thread< nbodies){
    double x = states[thread].x, y = states[thread].y;
    double vx = states[thread].vx, vy = states[thread].vy;
    // ax times delta t, ay times delta t...
    double ax_delta_t = 0.0, ay_delta_t = 0.0;

    for (int j=0; j<nbodies; j++) {
      if (j == thread) continue;
      
      const double dx = states[j].x - x, dy = states[j].y - y;
      const double mass = bodies[j].mass;
      const double r_squared = dx*dx + dy*dy;
      
      if (r_squared != 0) {
        const double r = sqrt(r_squared);

        if (r != 0) {
          const double acceleration = G * mass / r_squared;
          const double atOverr = acceleration * delta_t / r;
          
          ax_delta_t += dx * atOverr;
          ay_delta_t += dy * atOverr;
        }
      }
    }
    vx += ax_delta_t;
    vy += ay_delta_t;
    x += delta_t * vx;
    y += delta_t * vy;   
    assert(!isnan(x) && !isnan(y) && !isnan(vx) && !isnan(vy));
    states_new[thread] = (State){x, y, vx, vy};
  }
}

/* Close GIF file, free all allocated data structures */
static void wrapup() {
  ANIM_Close(af);
  free(posbuf);
  free(states);
  cudaFree(states_dev);
  cudaFree(states_dev_new);
  cudaFree(bodies_dev);
  printf("\nnbody: finished.  Time = %lf\n", ANIM_time() - start_time);
}

/* One argument: the name of the output file */
int main(int argc, char * argv[]) {
  int statbar = 0; // used for printing status updates
  assert(argc == 2);
  init(argv[1]);

  const int numBlocks = nbodies/ThreadsPerBlock + (nbodies%ThreadsPerBlock!=0);

  //update<<<numBlocks, ThreadsPerBlock>>>(states_dev, states_dev_new, bodies_dev, nbodies, delta_t);
  //cudaMemcpy(states, states_dev_new, nbodies*sizeof(State), cudaMemcpyDeviceToHost);
  if (wstep != 0) write_frame();
  for (int i=1; i<=nstep; i++) {
    //calculate the changes and place them in states_dev_new
    update<<<numBlocks, ThreadsPerBlock>>>(states_dev, states_dev_new, bodies_dev, nbodies, delta_t);
    ANIM_Status_update(stdout, nstep, i, &statbar);

    

    //swap the device state and state_new pointers
    State * const tmp = states_dev; states_dev = states_dev_new; states_dev_new = tmp;
    if (wstep != 0 && i%wstep == 0) {
      //copy the changes from the device back to the host
      //In my testing, doing this only when I needed to write didn't seem to have much of an impact
      //Might as well do it anyway though
      cudaMemcpy(states, states_dev_new, nbodies*sizeof(State), cudaMemcpyDeviceToHost);
      write_frame();
    }
  }
  wrapup();
}
