
#include <mpi.h>
#include <stdio.h>
#define N 10

struct Partstruct {
  int    type; /* particle type */
  double d[6]; /* particle coordinates */
  char   b[7]; /* some additional information */
};

int nprocs, rank;
MPI_Datatype Particletype; /* this type will correspond to the struct */

struct Partstruct particle[N];

void setup() {
  MPI_Datatype Particlestruct;
  MPI_Datatype type[3] = {MPI_INT, MPI_DOUBLE, MPI_CHAR};
  int blocklen[3] = {1, 6, 7};
  MPI_Aint disp[3];
  MPI_Aint base, sizeofentry;

  /* compute displacements of structure components */
  MPI_Get_address(particle, disp);
  MPI_Get_address(particle[0].d, disp+1);
  MPI_Get_address(particle[0].b, disp+2);
  base = disp[0];
  for (int i=0; i < 3; i++)
    disp[i] = MPI_Aint_diff(disp[i], base);
  MPI_Type_create_struct(3, blocklen, disp, type, &Particlestruct);
  /* compute extent of the structure */
  MPI_Get_address(particle+1, &sizeofentry);
  sizeofentry = MPI_Aint_diff(sizeofentry, base);
  /* build datatype describing structure */
  MPI_Type_create_resized(Particlestruct, 0, sizeofentry, &Particletype);
  MPI_Type_commit(&Particletype);
}

void print_particle(struct Partstruct * p) {
  printf("{%d, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %s}",
	 p->type, p->d[0], p->d[1], p->d[2], p->d[3], p->d[4], p->d[5],
	 p->b);
}

void print_all() {
  for (int i=0; i<N; i++) {
    print_particle(particle + i);
    printf("\n");
  }
  printf("\n");
}
    
void teardown() {
  MPI_Type_free(&Particletype);
}

int main() {
  const char str[7] = "abcdef";
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  setup();

  if (rank == 0) {
    for (int i=0; i<N; i++) {
      particle[i].type = i;
      for (int j=0; j<6; j++)
	particle[i].d[j] = j+1;
      for (int j=0; j<7; j++)
	particle[i].b[j] = str[j];
    }
    printf("Sending:\n");
    print_all();
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Send(particle, N, Particletype, 1, 0, MPI_COMM_WORLD);
  } else if (rank == 1) {
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Recv(particle, N, Particletype, 0, 0, MPI_COMM_WORLD, &status);
    printf("Received:\n");
    print_all();
  }
  teardown();
  MPI_Finalize();
}
