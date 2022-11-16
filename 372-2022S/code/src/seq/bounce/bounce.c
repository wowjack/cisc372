/* bounce.c: simple animation of ball bouncing around. */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "anim.h"

/* Global variables */
const int radius = 20;    /* radius of ball */
int nx, ny;               /* dimensions of the room (in pixels) */
double vx, vy;            /* velocity in x and y direction */
const double ay = -.2;    /* acceleration due to gravity */
ANIM_color_t color;       /* color of ball */
int nstep;                /* number of time steps */
int wstep;                /* time between writes to file */
char * filename;          /* name of file to create */
double pos[2];            /* position buffer: x and y coordinates */
ANIM_File af;             /* output file */
double start_time;        /* time simulation starts */

static void quit() {
  printf("Usage: bounce.exec NX NY VX NSTEP WSTEP FILENAME              \n\
  NX = number of pixels in x-direction                                  \n\
  NY = number of pixels in y-direction                                  \n\
  VX =  velocity in x direction (pixels per time step)                  \n\
  NSTEP = total number of time steps, at least 1                        \n\
  WSTEP = number of time steps between writes to file, in [0, NSTEP]    \n\
  FILENAME = name of output file                                        \n\
Example: bounce.exec 400 400 1 600 1 out.anim                           \n");
  exit(1);
}

static void setup(int argc, char * argv[]) {
  if (argc != 7) quit();
  nx = atoi(argv[1]), ny = atoi(argv[2]), vx = atof(argv[3]),
    nstep = atoi(argv[4]), wstep = atoi(argv[5]), filename = argv[6];
  if (!(nx>=2*radius && ny>=2*radius && nstep>=1 && wstep>=0 && wstep<=nstep))
    quit();
  printf("bounce: nx=%d ny=%d vx=%.3lf nstep=%d wstep=%d\n",
	 nx, ny, vx, nstep, wstep);
  const int nframes = wstep == 0 ? 0 : 1+nstep/wstep;
  printf("bounce: creating ANIM file %s with %d frames, %zu bytes.\n",
	 filename, nframes, ANIM_Nbody_file_size(2, 1, 1, nframes));
  fflush(stdout);
  start_time = ANIM_time();
  pos[0] = nx/2.;
  pos[1] = ny-radius-1;
  vy = 0;
  color = ANIM_Make_color(255, 255, 255);
  //  ANIM_Set_debug();
  af = ANIM_Create_nbody(2, (int[]){nx, ny},
			 (ANIM_range_t[]){{0, nx}, {0, ny}},
			 1, (int[]){radius}, 1, (ANIM_color_t[]){color},
			 (int[]){0}, filename);
}

static void teardown() {
  ANIM_Close(af);
  printf("\nbounce: finished.  Time = %lf\n", ANIM_time() - start_time);
}

void update() {
  pos[0] += vx;
  double dx = radius - pos[0];
  if (dx > 0) { // bounce off left wall
    pos[0] = radius + dx;
    vx = -vx;
  } else {
    dx = pos[0] - (nx - radius);
    if (dx > 0) { // bounce off right wall
      pos[0] = nx - radius - dx;
      vx = -vx;
    }
  }
  vy += ay;
  pos[1] += vy;
  if (pos[1] < radius) {
    vy = -0.8*vy;
    pos[1] = radius + (radius - pos[1]);
  }
}

int main(int argc, char *argv[]) {
  int dots = 0; // number of dots printed so far (0..100)

  setup(argc, argv);
  if (wstep != 0) ANIM_Write_frame(af, pos);
  for (int i = 1; i <= nstep; i++) {
    update();
    ANIM_Status_update(stdout, nstep, i, &dots);
    if (wstep != 0 && i%wstep == 0) ANIM_Write_frame(af, pos);
  }
  teardown();
}
