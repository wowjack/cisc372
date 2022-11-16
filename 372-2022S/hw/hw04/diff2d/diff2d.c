#include <stdlib.h>
#include <assert.h>
#include "anim.h"

const int n = 200;             // number of discrete points including endpoints
int nstep = 200000;            // number of time steps
int wstep = 400;               // time between writes to file
const double m = 100.0;        // initial temperature of rod interior
const int h0 = n/2 - 2, h1 = n/2 + 2;  // endpoints of heat source
const double k = 0.2;          // diffusivity constant
char * filename = "diff2d.anim";       // name of file to create
double ** u, ** u_new;         // two copies of the temperature function
ANIM_File af;                  // output file
double start_time;             // time simulation starts

static void setup(void) {
  start_time = ANIM_time();
  printf("diff2d: n=%d nstep=%d wstep=%d\n", n, nstep, wstep);
  u = ANIM_allocate2d(n,n);
  u_new = ANIM_allocate2d(n,n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      u_new[i][j] = u[i][j] = 0.0;
  for (int i = h0; i < h1; i++)
    for (int j = h0; j < h1; j++)
      u_new[i][j] = u[i][j] = m;
  af = ANIM_Create_heat
    (2, (int[]){n, n}, (ANIM_range_t[]){{0, 1}, {0, 1}, {0, m}}, filename);
}

static void teardown(void) {
  ANIM_Close(af);
  ANIM_free2d(u);
  ANIM_free2d(u_new);
  printf("\nTime (s) = %lf\n", ANIM_time() - start_time);
}

static void update() {
  for (int i = 1; i < n-1; i++)
    for (int j = 1; j < n-1; j++)
      u_new[i][j] =  u[i][j] +
	k*(u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4*u[i][j]);
  for (int i = 0; i < n;   i++) u_new[i][n-1] = u_new[i][n-2];
  for (int i = 0; i < n;   i++) u_new[i][0]   = u_new[i][1];
  for (int j = 1; j < n-1; j++) u_new[0][j]   = u_new[1][j];
  for (int j = 1; j < n-1; j++) u_new[n-1][j] = u_new[n-2][j];
  for (int i = h0; i < h1; i++)
    for (int j = h0; j < h1; j++)
      u_new[i][j] = m;
  double ** const tmp = u_new; u_new = u; u = tmp;
}

int main() {
  int dots = 0;
  setup();
  if (wstep != 0) ANIM_Write_frame(af, u[0]);
  for (int i = 1; i <= nstep; i++) {
    update();
    ANIM_Status_update(stdout, nstep, i, &dots);
    if (wstep != 0 && i%wstep == 0) ANIM_Write_frame(af, u[0]);
  }
  teardown();
}
