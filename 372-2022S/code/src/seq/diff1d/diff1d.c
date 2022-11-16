#include <stdlib.h>
#include <assert.h>
#include "anim.h"

/* Global variables */
const double m = 100.0;        // initial temperature of rod interior
const int n = 1200;            // number of discrete points including endpoints
const int h0 = n/2 - 2,        // left endpoint of heat source
  h1 = n/2 + 2;                // right endpoint of heat source
const double k = 0.05;         // D*dt/(dx*dx), diffusivity constant
int nstep = 10000000;          // number of time steps
int wstep = 20000;             // time between writes to file
char * filename =
  "diff1d.anim";               // name of file to create
double * u, * u_new;           // two copies of the temperature function
ANIM_File af;                  // output file
double start_time;             // time simulation starts

static void setup(void) {
  start_time = ANIM_time();
  u = (double*)malloc(n*sizeof(double));
  assert(u);
  u_new = (double*)malloc(n*sizeof(double));
  assert(u_new);
  for (int i = 0; i < n; i++) u_new[i] = u[i] = 0.0;
  for (int i = h0; i < h1; i++) u_new[i] = u[i] = m;
  af = ANIM_Create_heat(1, (int[]){n}, (ANIM_range_t[]){{0, 1}, {0, m}}, filename);
  //af = ANIM_Create_graph(1, (int[]){n,800}, (ANIM_range_t[]){{0,1}, {-1,m+1}}, filename);
}

static void teardown(void) {
  ANIM_Close(af);
  free(u);
  free(u_new);
}

static void update() {
  for (int i = 1; i < n-1; i++)
    u_new[i] =  u[i] + k*(u[i+1] + u[i-1] - 2*u[i]); // diffusion equation
  u_new[0] = u[0] + k*(u[1]-u[0]); // no flux at left boundary
  u_new[n-1] = u[n-1] + k*(u[n-2]-u[n-1]); // nor at right boundary
  for (int i = h0; i < h1; i++) u_new[i] = u[i] = m; // constant heat source
  double * const tmp = u_new; u_new = u; u = tmp;  // swap u and u_new
}

int main() {
  int dots = 0; // number of dots printed so far (0..100)
  setup();
  if (wstep != 0) ANIM_Write_frame(af, u);
  for (int i = 1; i <= nstep; i++) {
    update();
    ANIM_Status_update(stdout, nstep, i, &dots);
    if (wstep != 0 && i%wstep == 0) ANIM_Write_frame(af, u);
  }
  teardown();
}
