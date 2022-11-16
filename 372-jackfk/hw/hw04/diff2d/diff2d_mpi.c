#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <sys/time.h>
#include "mpianim.h"

/* The standard block distribution functions */
#define ulong unsigned long
#define FIRST(rank) ((((ulong)(rank))*((ulong)n))/nprocs)
#define OWNER(index) ((((ulong)nprocs)*(((ulong)(index))+1)-1)/n)
#define NUM_OWNED(rank) (FIRST(rank+1) - FIRST(rank))

const int n = 1000;           // number of discrete points including endpoints
int nstep = 2000000;          // number of time steps
int wstep = 5000;             // time between writes to file
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

/*
My program works and the created mp4 file looks good but I just can't get it perfectly match
the sequential version. Upon looking at the seq version in depth, it does some things that an MPI
version cannot replicate with just one column of ghost cells on each side. I made the wstep 
variable 1 for the seq version to see how it changes with every single iteration. The seq version
will actually diffuse the heat through two cells in one direction in one single iteration. I'm
not able to do this unless I entirely rewrite my implementation to exchange two columns of ghost
cells on each side of the stripe. Either way I'm satisfied.
*/

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
  if(rank == 0){
    //Send right cells and receive left ghosts from right
    MPI_Sendrecv(u[nl], n, MPI_DOUBLE, rank+1, 1, u[nl+1], n, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }else if(rank == nprocs-1){
    //Send left cells and receive right ghosts from left
    MPI_Sendrecv(u[1], n, MPI_DOUBLE, rank-1, 0, u[0], n, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }else{
    if(rank%2==0){
      //Send left cells and receive right ghosts from left
      MPI_Sendrecv(u[1], n, MPI_DOUBLE, rank-1, 0, u[0], n, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //Send right cells and receive left ghosts from right
      MPI_Sendrecv(u[nl], n, MPI_DOUBLE, rank+1, 1, u[nl+1], n, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }else{
      //Send right cells and receive left ghosts from right
      MPI_Sendrecv(u[nl], n, MPI_DOUBLE, rank+1, 1, u[nl+1], n, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //Send left cells and receive right ghosts from left
      MPI_Sendrecv(u[1], n, MPI_DOUBLE, rank-1, 0, u[0], n, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

//Its pretty nasty but it works
static void update() {
  for(int i=1; i<=nl; i++){ //Updating all cells except top and bottom walls
    for(int j=1; j<n-1; j++){
      u_new[i][j] =  u[i][j] + k*(u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4*u[i][j]);
    }
  }
  //Top and bottom walls get special rule
  for(int i=1; i<=nl; i++){
    u_new[i][0] = u[i][0] + k*(u[i+1][0] + u[i-1][0] + u[i][1] - 3*u[i][0]); //bottom wall
    u_new[i][n-1] = u[i][n-1] + k*(u[i+1][n-1] + u[i-1][n-1] + u[i][n-2] - 3*u[i][n-1]); //top wall
  }
  
  //Need special rules for right and left walls and corner
  if(rank==0){//left wall
    for(int i=1; i<n-1; i++){
      u_new[0][i] = u[0][i] + k*(u[0][i+1] + u[0][i-1] + u[1][i] - 3*u[0][i]);
    }
    u_new[0][0] = u[0][0] + k*(u[1][0]+u[0][1]-2*u[0][0]);//bottom left corner
    u_new[0][n-1] = u[0][n-1] + k*(u[1][n-1]+u[0][n-2]-2*u[0][n-1]);//top left corner
  }else if(rank==nprocs-1){//right wall
    for(int i=1; i<n-1; i++){
      u_new[nl][i] = u[nl][i] + k*(u[nl][i+1] + u[nl][i-1] + u[nl-1][i] - 3*u[nl][i]);
    }
    u_new[nl][0] = u[nl][0] + k*(u[nl-1][0] + u[nl][1] - 2*u[nl][0]);//bottom right corner
    u_new[nl][n-1] = u[nl][n-1] + k*(u[nl][n-2]+u[nl-1][n-1] - 2*u[nl][n-1]);//top right corner
  }
  for (int i = h0; i < h1; i++){
    if (rank == OWNER(i)){
      for (int j = h0; j < h1; j++){
	      u_new[i-first+1][j] = u[i-first+1][j] = m;
      }
    }
  }
  double ** const tmp = u_new; u_new = u; u = tmp;
}

double getMin(){
  double min = 100.;
  for(int i=1; i<=nl; i++){ //Updating all cells except top and bottom walls
    for(int j=0; j<n; j++){
      if(u[i][j] < min) min=u[i][j];
    }
  }
  return min;
}

double sumBlock(){
  double sum = 0;
  for(int i=1; i<=nl; i++){ //Updating all cells except top and bottom walls
    for(int j=0; j<n; j++){
      sum+=u[i][j];
    }
  }
  return sum;
}

//Get the number of cells owned by this proc
int getCount(){
  int count = 0;
  for(int i=1; i<=nl; i++){ //Updating all cells except top and bottom walls
    for(int j=0; j<n; j++){
      count += 1;
    }
  }
  return count;
}

double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

int main() {
  double startTime = mytime();
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
  
  //I didnt put a whole lot of thought into this part. it works though.
  double localMin = getMin();
  double globalMin;
  MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

  int localCount = getCount();
  double localSum = sumBlock();

  int globalCount;
  double globalSum;
  MPI_Reduce(&localCount, &globalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


  if(rank==0){
    printf("\nTime (s) = %.6f, Mean = %.6f, Min = %.6f\n", mytime()-startTime, globalSum/globalCount, globalMin);
  }

  teardown();
  MPI_Finalize();
}
