
#include <stdio.h>
#include <stdlib.h>

int nprocs;
int n_global;

// the first global index owned by process of rank r:
#define FIRST(r) (n_global*(r)/nprocs)

// the number of things owned by process r:
#define NUM_OWNED(r) (FIRST(r+1) - FIRST(r))

// the rank of the process that owns global index j:
#define OWNER(j) ((nprocs*((j)+1)-1)/n_global)

// the local index of the cell with global index j:
#define LOCAL_INDEX(j) ((j)-FIRST(OWNER(j)))


int main(int argc, char *argv[]) {
  n_global = atoi(argv[1]);
  nprocs = atoi(argv[2]);
  printf("Rank First NumLoc\n");
  printf("=================\n");
  for (int i=0; i<nprocs; i++) {
    printf("%4d %5d %6d\n", i, FIRST(i), NUM_OWNED(i));
  }
}
