#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include "mpianim.h"

/* The standard block distribution functions */
#define ulong unsigned long
#define FIRST(rank) ((((ulong)(rank))*((ulong)n))/nprocs)
#define OWNER(index) ((((ulong)nprocs)*(((ulong)(index))+1)-1)/n)
#define NUM_OWNED(rank) (FIRST(rank+1) - FIRST(rank))

const int n = 200;           // number of discrete points including endpoints
int nstep = 200000;          // number of time steps
int wstep = 400;             // time between writes to file
//const int n = 1000;         // number of discrete points including endpoints
//int nstep = 2000000;       // number of time steps
//int wstep = 5000;          // time between writes to file
const double m = 100.0;      // initial temperature of rod interior
const int h0 = n/2 - 2, h1 = n/2 + 2;  // endpoints of heat source
const double k = 0.2;        // D*dt/(dx*dx), diffusivity constant
char * filename = "diff2d_mpi.anim";   // name of file to create
double ** u, ** u_new;       // two copies of the temperature function
MPIANIM_File af;             // output file
double start_time;           // time simulation starts
int nprocs, rank;            // number of processes, rank of this process
int left, right;             // rank of left, right neighbor or MPI_PROC_NULL
int nl;                      // number of stripes "owned" by this proc
int first;                   // global index of first stripe owned by this proc

static void setup(void) {
  start_time = MPI_Wtime();
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    printf("diff2d_mpi: n=%d nstep=%d wstep=%d nprocs=%d\n",
	   n, nstep, wstep, nprocs);
  assert(n >= nprocs); // need at least one stripe per process
  first = FIRST(rank); 
  nl = NUM_OWNED(rank);
  left = rank > 0 ? rank - 1 : MPI_PROC_NULL;
  right = rank < nprocs - 1 ? rank + 1 : MPI_PROC_NULL;
  u = ANIM_allocate2d(nl+2, n);
  u_new = ANIM_allocate2d(nl+2, n);
  for (int i = 0; i < nl+2; i++)
    for (int j = 0; j < n; j++)
      u_new[i][j] = u[i][j] = 0.0;
  for (int i = h0; i < h1; i++)
    if (rank == OWNER(i))
      for (int j = h0; j < h1; j++)
	u_new[i-first+1][j] = u[i-first+1][j] = m;
  af = MPIANIM_Create_heat
    (2, (int[]){n, n}, (ANIM_range_t[]){{0, 1}, {0, 1}, {0, m}},
     (int[]){nl, n}, (int[]){first, 0}, filename, MPI_COMM_WORLD);
  MPIANIM_Set_nframes(af, (wstep == 0 ? 0 : 1+nstep/wstep));
}

static void teardown(void) {
  MPIANIM_Close(af);
  ANIM_free2d(u);
  ANIM_free2d(u_new);
  if (rank == 0)
    printf("\nTime (s) = %lf\n", MPI_Wtime() - start_time);
}

static void exchange_ghost_cells() {
  // TODO
}

static void update() {
  // TODO
}

int main() {
  int dots = 0;
  MPI_Init(NULL, NULL);
  setup();
  if (wstep != 0)
    MPIANIM_Write_frame(af, &u[1][0], MPI_STATUS_IGNORE);
  for (int i = 1; i <= nstep; i++) {
    exchange_ghost_cells();
    update();
    if (rank == 0)
      ANIM_Status_update(stdout, nstep, i, &dots);
    if (wstep != 0 && i%wstep == 0)
      MPIANIM_Write_frame(af, &u[1][0], MPI_STATUS_IGNORE);
  }
  teardown();
  MPI_Finalize();
}
