
#include <mpi.h>
#include "mpianim.h"

int nprocs, rank;

int main() {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ANIM_Set_debug();
  MPIANIM_File af = MPIANIM_Create_heat(1, (int[]){nprocs}, (ANIM_range_t[]){{0,100},{0,100}},
					(int[]){1}, (int[]){rank}, "test1.anim", MPI_COMM_WORLD);
  double x = rank;

  MPIANIM_Write_frame(af, &x, MPI_STATUS_IGNORE);
  MPIANIM_Close(af);
  MPI_Finalize();
}
