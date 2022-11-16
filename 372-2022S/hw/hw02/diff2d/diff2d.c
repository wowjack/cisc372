#include <stdlib.h>
#include <assert.h>
#include "anim.h"

/* Global variables */
const double m = 100.0;        // initial temperature of rod interior
const int n = 200;             // start with something smaller, then scale up
const int h0 = n/2 - 2,        // left endpoint of heat source
  h1 = n/2 + 2;                // right endpoint of heat source
const double k = 0.2;          // D*dt/(dx*dx), diffusivity constant
int nstep = 200000;            // number of time steps: again, start small
int wstep = 400;               // time between writes to file: ditto
char * filename =
  "diff2d.anim";               // name of file to create
// TODO: complete the variable decls


static void setup(void) {
  // TODO
}

static void teardown(void) {
  // TODO
}

static void update() {
  for (int i = 1; i < n-1; i++)
    for (int j = 1; j < n-1; j++)
      u_new[i][j] =  u[i][j] +
	k*(u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4*u[i][j]);
  // TODO: update edges, corners, center square, ..., what else?
}

int main() {
  int dots = 0; // number of dots printed so far (0..100)
  setup();
  if (wstep != 0) ANIM_Write_frame(af, u[0]);
  for (int i = 1; i <= nstep; i++) {
    update();
    ANIM_Status_update(stdout, nstep, i, &dots);
    if (wstep != 0 && i%wstep == 0) ANIM_Write_frame(af, /* WHAT GOES HERE? */);
  }
  teardown();
}
