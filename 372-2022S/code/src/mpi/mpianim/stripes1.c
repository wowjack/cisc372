/* Simple illustration of the MPIANIM library.  Each proc gets a
   vertical stripe of the rectangular domain.  The nx x ny global
   domain is partitioned into vertical stripes, one stripe for each
   process.  Process rank is responsible for all x-coordinates from
   first_x to first_x+nxl-1.  Note that lengths={nx,ny} are the global
   dimensions.  Argument sublengths = {nxl,ny} gives the dimensions of
   the stripe belonging to this process.  starts={first_x,0} is the
   coordinate of the lower left corner of the stripe belonging to this
   proc.  The choice of ranges specifies that the x-axis will be
   labeled from 0 to 1, and same for the y-axis (this data is
   currently not used).  The value taken on by the function at any
   point must be a double in the range [0,nprocs-1].

   In this simple example, a single animation frame is created, and
   each proc sets its region of the domain to the rank of the proc.
   These will appear as different colors in the blue-red spectrum,
   with pure blue corresonding to the lowest rank (i.e., 0).
 */
#include <stdlib.h>
#include <assert.h>
#include "mpianim.h"

int rank, nprocs;
MPIANIM_File af;
int nx = 800, ny = 400;
int nxl;
int first_x;
char * filename = "stripes1.anim";
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
  for (int i=0; i<nxl; i++)
    for (int j=0; j<ny; j++)
      buf[i*ny + j] = rank;
  MPIANIM_Write_frame(af, buf, MPI_STATUS_IGNORE);
  free(buf);
  MPIANIM_Close(af);
  MPI_Finalize();
}
