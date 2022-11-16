#include <mpi.h>
#include <stdio.h>
#define INTERVALS 1000000000L // 10^9 = 1 billion

int main () {
  const long double delta = 1.0L/(long double)INTERVALS,
    delta4 = 4.0L*delta;
  long double my_area = 0.0L, total_area;
  int rank, nprocs;
  double start_time;

  MPI_Init(NULL, NULL);
  start_time = MPI_Wtime();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  for (long i = rank; i < INTERVALS; i += nprocs) {
    const long double x = ((long double)i + 0.5L)*delta;
    my_area += delta4/(1.0L + x*x);
  }
  MPI_Reduce(&my_area, &total_area, 1, MPI_LONG_DOUBLE, MPI_SUM, 0,
	     MPI_COMM_WORLD);
  if (rank == 0)
    printf("Pi is %20.17Lf.  Time = %lf\n", total_area,
	   MPI_Wtime() - start_time);
  MPI_Finalize();
}
