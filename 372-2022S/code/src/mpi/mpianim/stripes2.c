/* Each proc gets a horizontal stripe */
#include <stdlib.h>
#include <assert.h>
#include "mpianim.h"

int rank, nprocs;
MPIANIM_File af;
int nx = 800, ny = 400;
int nyl;
int first_y;
char * filename = "stripes2.anim";
double * buf;

int main() {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  nyl = ny/nprocs;
  first_y = rank*nyl;
  if (rank == nprocs - 1) nyl += ny%nprocs;
  af = MPIANIM_Create_heat(2, (int[]){nx, ny},
			   (ANIM_range_t[]){{0, 1}, {0, 1}, {0, nprocs-1}},
			   (int[]){nx, nyl}, (int[]){0, first_y}, filename,
			   MPI_COMM_WORLD);
  buf = malloc(nx * nyl * sizeof(double));
  assert(buf);
  for (int i=0; i<nx; i++)
    for (int j=0; j<nyl; j++)
      buf[i*nyl + j] = rank;
  MPIANIM_Write_frame(af, buf, MPI_STATUS_IGNORE);
  free(buf);
  MPIANIM_Close(af);
  MPI_Finalize();
}
