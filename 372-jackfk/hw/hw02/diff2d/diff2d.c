#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include "anim.h"

/* Global variables */
const double m = 100.0;        // initial temperature of rod interior
const int n = 18;             // start with something smaller, then scale up
const double k = 0.2;          // D*dt/(dx*dx), diffusivity constant
int nstep = 2000000;            // number of time steps: again, start small
int wstep = 400;               // time between writes to file: ditto
char * filename =
  "diff2d.anim";               // name of file to create
// TODO: complete the variable decls

//////////////////////////////////MY CODE
double** u_new;
double** u;
ANIM_File af;

//I changed the names here because h0 & h1 arent exactly intuitive
//Making top and bottom variables here isn't exactly required considering the 2d array is square,
//I just did it to make indexing using these variables more readable
//Note that this 
const int hleft = n/2-2,        // left endpoint of heat source
  hright = n/2+2,               // right endpoint of heat source
  htop = hleft,                   // top endpoint of heat source
  hbottom = hright;               // bottom endpoint of the heat source


//My method to heap allocate arrays
static void allocateArrays(){
  //Heap allocate arrays of size n of pointer to pointer to double
  u = (double**)malloc(n*sizeof(double*));  
  u_new = (double**)malloc(n*sizeof(double*));
  //Heap allocate arrays of doubles of size n and store the pointer in the previously allocated arrays
  for(int i=0; i<n; i++){
    u[i] = (double*)malloc(n*sizeof(double));
    u_new[i] = (double*)malloc(n*sizeof(double));
  }
}

//My method to print the 2d array
void printarr(double** a, int ySize, int xSize){
  for(int y=0; y<ySize; y++){
    for(int x=0; x<xSize; x++){
      printf("%.2f\t", a[y][x]);
    }
    printf("\n");
  }
}

static void setHeatCells(double** arr){
  for(int y=htop; y<hbottom; y++){
    for(int x=hleft; x<hright; x++){
      arr[y][x] = 100;
    }
  }
}

static void setup(void) {
  // TODO
  allocateArrays();
  //Set the heat cells value
  setHeatCells(u);
  //Not really sure what these asserts are doing
  //They're in the 1d example so ill just put them here too
  assert(u);
  assert(u_new);

  af = ANIM_Create_heat(2, (int[]){n,n}, (ANIM_range_t[]){{0, 1}, {0,1}, {0, m}}, filename);

}

static void teardown(void) {
  // TODO
  for(int i=0; i<n; i++){
    free(u[i]);
    free(u_new[i]);
  }
  free(u);
  free(u_new);
}

static void update() {
  for (int i = 1; i < n-1; i++){
    for (int j = 1; j < n-1; j++){
      u_new[i][j] =  u[i][j] + k*(u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4*u[i][j]);
    }
  }
  //make sure the center cells stay at 100
  setHeatCells(u_new);
  //Update the corners
  u_new[0][0] = u[0][0] + k*(u[1][0] + u[0][1] - 2*u[0][0]); //top left
  u_new[n-1][0] = u[n-1][0] + k*(u[n-2][0] + u[n-1][1] - 2*u[n-1][0]); //bottom left
  u_new[0][n-1] = u[0][n-1] + k*(u[0][n-2] + u[1][n-1] - 2*u[0][n-1]); //top right
  u_new[n-1][n-1] = u[n-1][n-1] + k*(u[n-2][n-1] + u[n-1][n-2] - 2*u[n-1][n-1]); //bottom right
  //update the edges
  for(int i=1; i<n-1; i++){
    //left edge
    u_new[i][0] = u[i][0] + k*(u[i-1][0] + u[i+1][0] + u[i][1] - 3*u[i][0]);
    //right edge
    u_new[i][n-1] = u[i][n-1] + k*(u[i-1][n-1] + u[i+1][n-1]+ u[i][n-2] - 3*u[i][n-1]);
    //top edge
    u_new[0][i] = u[0][i] + k*(u[0][i-1] + u[0][i+1] + u[1][i] - 3*u[0][i]);
    //bottom edge
    u_new[n-1][i] = u[n-1][i] + k*(u[n-1][i-1] + u[n-1][i+1] + u[n-2][i] - 3*u[n-1][i]);
  }
  //swap the u and u_new pointers
  double** tmp = u_new;
  u_new = u;
  u = tmp;
}

int main() {
  /*
  setup();
  for(int i=1; i<nstep; i++){
    update();
    printf("\n\n\n\n\n\n\n\n");
    printarr(u, n, n);
    usleep(100000);
  }
  teardown();
  */

  int dots = 0; // number of dots printed so far (0..100)
  setup();
  if (wstep != 0) ANIM_Write_frame(af, u[0]);
  for (int i = 1; i <= nstep; i++) {
    update();
    ANIM_Status_update(stdout, nstep, i, &dots);
    if (wstep != 0 && i%wstep == 0) ANIM_Write_frame(af, (double*)u); //I just casted to double* to get rid of warning lol
  }
  teardown();
}
