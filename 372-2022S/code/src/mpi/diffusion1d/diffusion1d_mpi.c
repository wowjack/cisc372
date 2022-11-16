/* Name   : diffusion1d_mpi.c.  MPI version of diffusion1d.c.
   Author : Stephen F. Siegel, University of Delaware, 2020.
   The nx elements of the rod (including the two boundary cells) are
   distributed over nprocs processes.  Each process "owns" a block of
   nxl contiguous cells, plus two "ghost" cells. */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <mpi.h>
#include "mpianim.h"
/* The standard block distribution functions */
#define ulong unsigned long
#define FIRST(rank) ((((ulong)(rank))*((ulong)nx))/nprocs)
#define OWNER(index) ((((ulong)nprocs)*(((ulong)(index))+1)-1)/nx)
#define NUM_OWNED(rank) (FIRST(rank+1) - FIRST(rank))

/* Global variables */
const double m = 100.0;   /* initial temperature of rod interior */
int nx;                   /* global number of discrete points incl. boundary */
double k;                 /* D*dt/(dx*dx) */
int nstep;                /* total number of time steps */
int wstep;                /* number of time steps between writes to file */
char * filename;          /* name of output file */
double * u, * u_new;      /* two copies of temp. function, length nxl+2 */
MPIANIM_File af;          /* output file */
double start_time;        /* time simulation starts */
int nprocs, rank;         /* number of processes, rank of this process */
int left, right;          /* rank of left, right neighbor or MPI_PROC_NULL */
int nxl;                  /* number of cells "owned" by this proc */
int start, stop;          /* first and last local index to update */

/* Prints a usage message and exit */
static void quit() {
  printf("Usage: mpiexec -n NP diffusion1d_mpi.exec NX K NSTEPS WSTEP FILENAME \n\
  NX = number of points in rod, including the two endpoints           \n\
  K = D*dt/(dx*dx), a constant conrolling rate of diffusion in (0,.5) \n\
  NSTEP = total number of time steps, at least 1                      \n\
  WSTEP = number of time steps between writes to file, in [0, NSTEP]  \n\
  FILENAME = name of output file                                      \n\
Example: mpiexec -n 4 diffusion1d_mpi.exec 100 0.3 1000 10 out.anim   \n");
  fflush(stdout);
  exit(1);
}

/* Parses command line args and initializes global variables. */
static void setup(int argc, char * argv[]) {
  int filename_length; // length of filename string (excluding null terminator)
  int nframes; // number of frames in the animation
  int first; // global index of first cell owned by this proc

  MPI_Barrier(MPI_COMM_WORLD); // for timing
  start_time = MPI_Wtime();
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (argc != 6) quit();
    nx = atoi(argv[1]), k = atof(argv[2]), nstep = atoi(argv[3]),
      wstep = atoi(argv[4]), filename = argv[5];
    if (!(nx>=2 && 0<k && k<.5 && nstep>=1 && wstep>=0 && wstep<=nstep))
      quit();
    nframes = wstep == 0 ? 0 : 1+nstep/wstep;
    filename_length = strlen(filename);
    printf("diffusion1d_mpi: nx=%d k=%lf nstep=%d wstep=%d nprocs=%d\n",
	   nx, k, nstep, wstep, nprocs);
    printf("diffusion1d_mpi: creating ANIM file %s with %d frames, %zu bytes.\n",
	   filename, nframes, ANIM_Heat_file_size(1, &nx, nframes));
    fflush(stdout);
  }
  MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&k, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nstep, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&wstep, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nframes, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&filename_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    filename = malloc(filename_length + 1);
    assert(filename);
  }
  MPI_Bcast(filename, filename_length + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
  first = FIRST(rank); 
  nxl = NUM_OWNED(rank);
  left = (first == 0 || nxl == 0 ? MPI_PROC_NULL : OWNER(first - 1));
  right = (first + nxl >= nx || nxl == 0 ? MPI_PROC_NULL : OWNER(first + nxl));
  u = malloc((nxl+2)*sizeof(double));
  assert(u);
  u_new = malloc((nxl+2)*sizeof(double));
  assert(u_new);
  if (rank == OWNER(0)) { // the left-most proc owning at least one cell
    u[1] = u_new[1] = 0.0;  // initialize left boundary in both copies of u
    start = 2; // u[2] is the first cell to be updated
  } else {
    start = 1; // u[1] is the first cell to be updated
  }
  if (rank == OWNER(nx-1)) { // the right-most proc owning at least one cell
    u[nxl] = u_new[nxl] = 0.0; // initialize the right boundary in both copies
    stop = nxl - 1; // u[nxl-1] is the last cell to be updated
  } else {
    stop = nxl; // u[nxl] is the last cell to be updated
  }
  for (int i = start; i <= stop; i++) u[i] = m;
#ifdef DEBUG
  printf("Proc %d: nx=%d, nxl=%d, first=%d\n", rank, nx, nxl, first);
  fflush(stdout);
  ANIM_Set_debug();
#endif
  af = MPIANIM_Create_heat(1, &nx, (ANIM_range_t[]){{0.0, 1.0}, {0.0, 100.0}},
			   &nxl, &first, filename, MPI_COMM_WORLD);
  MPIANIM_Set_nframes(af, nframes); // not required, but can improve IO
}

/* Frees allocated memory, closes the file, prints the time */
static void teardown() {
  MPIANIM_Close(af);
  free(u);
  free(u_new);
  MPI_Barrier(MPI_COMM_WORLD); // for timing
  if (rank == 0)
    printf("\ndiffusion1d_mpi: finished.  Time = %lf\n",
	   MPI_Wtime() - start_time);
  else
    free(filename);
}

/* Updates ghost cells using MPI communication */
static void exchange_ghost_cells() {
  MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, left, 0,
	       &u[nxl+1], 1, MPI_DOUBLE, right, 0,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&u[nxl], 1, MPI_DOUBLE, right, 0,
	       &u[0], 1, MPI_DOUBLE, left, 0,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/* Updates interior cells.  Requires ghost cells be up-to-date. */
void update() {
  for (int i = start; i <= stop; i++)
    u_new[i] = u[i] + k*(u[i+1] + u[i-1] - 2*u[i]);
  double * const tmp = u_new; u_new = u; u = tmp;
}

/* Runs the simulation.  See function quit() for args */
int main(int argc, char *argv[]) {
  int dots = 0; // number of dots printed so far (0..100)

  MPI_Init(&argc, &argv);
  setup(argc, argv);
  if (wstep != 0) MPIANIM_Write_frame(af, u+1, MPI_STATUS_IGNORE);
  for (int i = 1; i <= nstep; i++) {
    exchange_ghost_cells();
    update();
    if (rank == 0) ANIM_Status_update(stdout, nstep, i, &dots);
    if (wstep != 0 && i%wstep == 0)
      MPIANIM_Write_frame(af, u+1, MPI_STATUS_IGNORE);
  }
  teardown();
  MPI_Finalize();
}
