/* diffuse1d.c: sequential 1d Diffusion simulation with textual output.
   The length of the rod is 1. The endpoints are frozen at 0 degrees.
*/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

/* Global variables */
const double m = 100.0;   /* initial temperature of rod interior */
int nx;                   /* number of discrete points including endpoints */
double k;                 /* D*dt/(dx*dx) */
int nstep;                /* number of time steps */
int wstep;                /* time between writes to file */
double * u, * u_new;      /* two copies of temperature function */
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
  u_new = (double*)malloc(nx*sizeof(double));
  assert(u_new);
  for (int i = 1; i < nx - 1; i++) u[i] = m;
  u[0] = u_new[0] = u[nx-1] = u_new[nx-1] = 0.0;
}

static void teardown() {
  free(u);
  free(u_new);
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

static void update() {
  for (int i = 1; i < nx-1; i++)
    u_new[i] =  u[i] + k*(u[i+1] + u[i-1] - 2*u[i]);
  double * tmp = u_new; u_new = u; u = tmp;
}

int main(int argc, char *argv[]) {
  setup(argc, argv);
  if (wstep != 0) write(0);
  for (int i = 1; i <= nstep; i++) {
    update();
    if (wstep != 0 && i%wstep == 0)
      write(i);
  }
  teardown();
}
