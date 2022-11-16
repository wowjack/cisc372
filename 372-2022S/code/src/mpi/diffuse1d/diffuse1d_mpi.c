/* Name   : diffuse1d_mpi.c.  MPI version of diffuse1d.c.
   Author : Stephen F. Siegel, University of Delaware, 2020.
   The nx elements of the rod (including the two boundary cells) are
   distributed over nprocs processes.  Each process "owns" a block of
   nxl contiguous cells, plus two "ghost" cells.  */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <mpi.h>
/* The standard block distribution functions */
#define ulong unsigned long
#define FIRST(rank) ((((ulong)(rank))*((ulong)nx))/nprocs)
#define OWNER(index) ((((ulong)nprocs)*(((ulong)(index))+1)-1)/nx)
#define NUM_OWNED(rank) (FIRST(rank+1) - FIRST(rank))

/* Global variables */
const double m = 100.0;   /* initial temperature of rod interior */
int nx;                   /* global number of discrete points incl. boundary */
double k;                 /* D*dt/(dx*dx) */
int nstep;                /* number of time steps */
int wstep;                /* time between writes to file */
double * u, * u_new;      /* two copies of temperature function */
double start_time;        /* time simulation starts */
FILE * out = NULL;        /* where the output goes, used on proc 0 only */
int nprocs, rank;         /* number of processes, rank of this process */
int left, right;          /* rank of left, right neighbor or MPI_PROC_NULL */
int nxl;                  /* number of cells "owned" by this proc */
int start, stop;          /* first and last local index to update */

static void quit() {
  printf("Usage: mpiexec -n NP diffuse1d_mpi.exec NX K NSTEPS WSTEP [FILENAME] \n\
  NX = number of points in rod, including the two endpoints           \n\
  K = D*dt/(dx*dx), a constant conrolling rate of diffusion in (0,.5) \n\
  NSTEPS = total number of time steps                                 \n\
  WSTEP = number of time steps between writes to file                 \n\
  FILENAME = file to send output to (optional)                        \n\
Example: mpiexec -n 4 diffuse1d.exec 100 0.3 1000 10                  \n");
  exit(1);
}

static void setup(int argc, char * argv[]) {
  MPI_Barrier(MPI_COMM_WORLD); // start clock once everyone is here
  start_time = MPI_Wtime();
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (argc < 5 ||argc > 6) quit();
    nx = atoi(argv[1]), k = atof(argv[2]), nstep = atoi(argv[3]),
      wstep = atoi(argv[4]);
    out = argc == 5 ? stdout : fopen(argv[5], "w");
    assert(out);
    if (!(nx>=2 && 0<k && k<.5 && nstep>=1 && wstep>=0 && wstep<=nstep))
      quit();
    printf("diffuse1d_mpi: nx=%d k=%lf nstep=%d wstep=%d\n",
	   nx, k, nstep, wstep);
    fflush(stdout);
  }
  MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&k, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nstep, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&wstep, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int first = FIRST(rank); // global index of first cell owned by this proc

  nxl = NUM_OWNED(rank);
  left = (first == 0 || nxl == 0 ? MPI_PROC_NULL : OWNER(first - 1));
  right = (first + nxl >= nx || nxl == 0 ? MPI_PROC_NULL : OWNER(first + nxl));
  u = malloc((nxl+2)*sizeof(double)), u_new = malloc((nxl+2)*sizeof(double));
  assert(u); assert(u_new);
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
  //printf("Proc %d: nx=%d, nxl=%d, first=%d\n", rank, nx, nxl, first);
}

static void teardown() {
  free(u);
  free(u_new);
  MPI_Barrier(MPI_COMM_WORLD); // for timing
  if (rank == 0) {
    if (out != stdout) fclose(out);
    printf("\ndiffuse1d_mpi: finished.  Time = %lf\n",
	   MPI_Wtime() - start_time);
  }
}

static void write(int time) {
  if (rank == 0) {
    fprintf(out, "%4d: ", time);
    for (int i = 1; i <= nxl; i++)
      fprintf(out, "%7.2lf ", u[i]);
    for (int p = 1; p < nprocs; p++) {
      const int count = NUM_OWNED(p);

      MPI_Recv(u_new, count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      for (int i = 0; i < count; i++)
	fprintf(out, "%7.2lf ", u_new[i]);
    }
    fprintf(out, "\n"); fflush(out);
    u_new[1] = u_new[nxl] = 0.0;
  } else {
    MPI_Send(u + 1, nxl, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}

/* exchange_ghost_cells: updates ghost cells using MPI communication */
static void exchange_ghost_cells() {
  MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, left, 0,
	       &u[nxl+1], 1, MPI_DOUBLE, right, 0,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&u[nxl], 1, MPI_DOUBLE, right, 0,
	       &u[0], 1, MPI_DOUBLE, left, 0,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

static void update() {
  for (int i = start; i <= stop; i++)
    u_new[i] =  u[i] + k*(u[i+1] + u[i-1] - 2*u[i]);
  double * tmp = u_new; u_new = u; u = tmp;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  setup(argc, argv);
  if (wstep != 0) write(0);
  for (int i = 1; i <= nstep; i++) {
    exchange_ghost_cells();
    update();
    if (wstep != 0 && i%wstep == 0)
      write(i);
  }
  teardown();
  MPI_Finalize();
}
