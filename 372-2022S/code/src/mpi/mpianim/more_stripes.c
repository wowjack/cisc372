#include <stdlib.h>
#include <assert.h>
#include "mpianim.h"
#define FIRST(r) ((N)*(r)/nprocs)
#define NUM_OWNED(r) (FIRST((r)+1) - FIRST(r))
#define OWNER(j) ((nprocs*((j)+1)-1)/(N))
#define LOCAL_INDEX(j) ((j)-FIRST(OWNER(j)))

int rank, nprocs, N, ny = 800, nxl, first_x;
char * filename = "more_stripes.anim";

int main(int argc, char * argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  N = atoi(argv[1]);
  nxl = NUM_OWNED(rank);
  
  double a[nxl], buf[nxl*ny];
  
  first_x = FIRST(rank);
  MPIANIM_File af =
    MPIANIM_Create_heat(2, (int[]){N, ny},
			(ANIM_range_t[]){{0, 1}, {0, 1}, {0, N-1}},
			(int[]){nxl, ny}, (int[]){first_x, 0}, filename,
			MPI_COMM_WORLD);
  for (int i=0; i<nxl; i++)
    a[i] = first_x + i;
  for (int i=0; i<nxl; i++)
    for (int j=0; j<ny; j++)
      buf[i*ny + j] = a[i];
  MPIANIM_Write_frame(af, buf, MPI_STATUS_IGNORE);
  MPIANIM_Close(af);
  MPI_Finalize();
}
