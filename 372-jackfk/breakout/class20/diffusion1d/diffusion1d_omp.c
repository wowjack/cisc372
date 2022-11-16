/* Name   : diffusion1d.c.
   Author : Stephen F. Siegel, University of Delaware

   Sequential version of 1d diffusion, creating an animation in ANIM
   format.  The length of a metal rod is 1. The endpoints are frozen
   at 0 degrees (fixed boundary condition).  The interior points
   starts off at 100 degrees, and their temperatures change over time
   due to heat diffusion.  See function quit for usage.
*/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include "anim.h"
 
/* Global variables */
const double m = 100.0;   /* initial temperature of rod interior */
int nx;                   /* number of discrete points including endpoints */
double k;                 /* D*dt/(dx*dx) */
int nstep;                /* number of time steps */
int wstep;                /* time between writes to file */
char * filename;          /* name of file to create */
double * u, * u_new;      /* two copies of the temperature function */
ANIM_File af;             /* output file */
double start_time;        /* time simulation starts */

static void quit() {
  printf("Usage: diffusion1d.exec NX K NSTEPS WSTEP FILENAME          \n\
  NX = number of points in rod, including the two endpoints           \n\
  K = D*dt/(dx*dx), a constant conrolling rate of diffusion in (0,.5) \n\
  NSTEP = total number of time steps, at least 1                      \n\
  WSTEP = number of time steps between writes to file, in [0, NSTEP]  \n\
  FILENAME = name of output file                                      \n\
Example: diffusion1d.exec 100 0.3 1000 10 out.anim                    \n");
  exit(1);
}

static void setup(int argc, char * argv[]) {
  if (argc != 6) quit();
  nx = atoi(argv[1]), k = atof(argv[2]), nstep = atoi(argv[3]),
    wstep = atoi(argv[4]), filename = argv[5];
  if (!(nx>=2 && 0<k && k<.5 && nstep>=1 && wstep>=0 && wstep<=nstep))
    quit();
  printf("diffusion1d: nx=%d k=%lf nstep=%d wstep=%d.\n",
	 nx, k, nstep, wstep);
  const int nframes = wstep == 0 ? 0 : 1+nstep/wstep;
  printf("diffusion1d: creating ANIM file %s with %d frames, %zu bytes.\n",
	 filename, nframes, ANIM_Heat_file_size(1, &nx, nframes));
  fflush(stdout);
  start_time = ANIM_time();
  u = (double*)malloc(nx*sizeof(double));
  assert(u);
  u_new = (double*)malloc(nx*sizeof(double));
  assert(u_new);
  for (int i = 1; i < nx - 1; i++) u[i] = m;
  u[0] = u_new[0] = u[nx-1] = u_new[nx-1] = 0.0;
  af = ANIM_Create_heat(1, &nx,	(ANIM_range_t[]){{0.0, 1.0}, {0.0, 100.0}},
			filename);
}

static void teardown() {
  ANIM_Close(af);
  free(u);
  free(u_new);
  printf("\ndiffusion1d: finished.  Time = %lf\n", ANIM_time() - start_time);
}

static void update() {
  #pragma omp parallel for shared(u, u_new) schedule(dynamic, 1)
  for (int i = 1; i < nx-1; i++)
    u_new[i] =  u[i] + k*(u[i+1] + u[i-1] - 2*u[i]);
  double * const tmp = u_new; u_new = u; u = tmp;
}

int main(int argc, char *argv[]) {
  int dots = 0; // number of dots printed so far (0..100)

  setup(argc, argv);
  if (wstep != 0) ANIM_Write_frame(af, u);
  for (int i = 1; i <= nstep; i++) {
    update();
    ANIM_Status_update(stdout, nstep, i, &dots);
    if (wstep != 0 && i%wstep == 0) ANIM_Write_frame(af, u);
  }
  teardown();
}
