/* diffusion2d.c: simple 2d version of diffusion */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "anim.h"

/* Global variables */
int nx, ny;               /* dimensions of the room (in pixels) */
double k;                 /* constant controlling rate of diffusion */
int nstep;                /* number of time steps */
int wstep;                /* time between writes to file */
char * filename;          /* name of file to create */
double ** u, ** u_new;    /* two copies of temperature function */
ANIM_File af;             /* output file */
double start_time;        /* time simulation starts */

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
  nx = atoi(argv[1]), ny = atoi(argv[2]), k = atof(argv[3]),
    nstep = atoi(argv[4]), wstep = atoi(argv[5]), filename = argv[6];
  if (!(nx>=6 && ny>=6 && 0<k && k<.25 && nstep>=1 && wstep>=0 && wstep<=nstep))
    quit();
  printf("diffusion2d: nx=%d ny=%d k=%.3lf nstep=%d wstep=%d\n",
	 nx, ny, k, nstep, wstep);
  const int nframes = wstep == 0 ? 0 : 1+nstep/wstep;
  printf("diffusion2d: creating ANIM file %s with %d frames, %zu bytes.\n",
	 filename, nframes, ANIM_Heat_file_size(2, (int[]){nx, ny}, nframes));
  fflush(stdout);
  start_time = ANIM_time();
  u = ANIM_allocate2d( nx, ny );
  u_new = ANIM_allocate2d( nx, ny );
  for (int i = 1; i < nx - 1; i++)
    for (int j = 1; j < ny - 1; j++)
      u_new[i][j] = u[i][j] = 100.0;
  for (int i=0; i < nx; i++)
    u_new[i][0] = u[i][0] = u_new[i][ny-1] = u[i][ny-1] = 0.0;
  for (int j=1; j < ny - 1; j++)
    u_new[0][j] = u[0][j] = u_new[nx-1][j] = u[nx-1][j] = 0.0;
  af = ANIM_Create_heat(2, (int[]){nx, ny},
			(ANIM_range_t[]){{0, nx}, {0, ny}, {0, 100.0}},
			filename);
}

static void teardown() {
  ANIM_Close(af);
  ANIM_free2d(u);
  ANIM_free2d(u_new);
  printf("\ndiffusion2d: finished.  Time = %lf\n", ANIM_time() - start_time);
}

void update() {
  for (int i = 1; i < nx - 1; i++)
    for (int j = 1; j < ny - 1; j++)
      u_new[i][j] = u[i][j] +
	k*(u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4*u[i][j]);
  double ** const tmp = u_new; u_new = u; u = tmp;
}

int main(int argc, char *argv[]) {
  int dots = 0; // number of dots printed so far (0..100)

  setup(argc, argv);
  if (wstep != 0) ANIM_Write_frame(af, u[0]);
  for (int i = 1; i <= nstep; i++) {
    update();
    ANIM_Status_update(stdout, nstep, i, &dots);
    if (wstep != 0 && i%wstep == 0) ANIM_Write_frame(af, u[0]);
  }
  teardown();
}
