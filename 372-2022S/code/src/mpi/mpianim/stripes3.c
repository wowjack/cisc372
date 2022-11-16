/* Each proc gets a vertical stripe, fades to blue. */
#include <stdlib.h>
#include <assert.h>
#include "mpianim.h"

const int nstep = 120;
int rank, nprocs;
MPIANIM_File af;
int nx = 800, ny = 400;
int nxl;
int first_x;
char * filename = "stripes3.anim";
double * buf;

int main() {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  nxl = nx/nprocs;
  first_x = rank*nxl;
  if (rank == nprocs - 1) nxl += nx%nprocs;
  af = MPIANIM_Create_heat(2, (int[]){nx, ny},
			   (ANIM_range_t[]){{0, 1}, {0, 1}, {0, nprocs-1}},
			   (int[]){nxl, ny}, (int[]){first_x, 0}, filename,
			   MPI_COMM_WORLD);
  buf = malloc(nxl * ny * sizeof(double));
  assert(buf);
  for (int t=0; t<=nstep; t++) {
    for (int i=0; i<nxl; i++)
      for (int j=0; j<ny; j++)
	buf[i*ny + j] = 1.0*rank*(nstep-t)/nstep;
    MPIANIM_Write_frame(af, buf, MPI_STATUS_IGNORE);
  }
  free(buf);
  MPIANIM_Close(af);
  MPI_Finalize();
}
