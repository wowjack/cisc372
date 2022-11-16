/* perfect.c: find all perfect numbers up to a bound */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

_Bool is_perfect(int n) {
  if (n < 2) return 0;
  int sum = 1, i = 2;
  while (1) {
    const int i_squared = i*i;
    if (i_squared < n) {
      if (n%i == 0) sum += i + n/i;
      i++;
    } else {
      if (i_squared == n) sum += i;
      break;
    }
  }
  return sum == n;
}

int main(int argc, char * argv[]) {
  MPI_Init(&argc, &argv);
  int num_perfect = 0;
  double start_time = mytime();

  //Get the rank and number of procs
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  //Explain usage
  if (argc != 2 && rank==0) {
    printf("Usage: perfect.exec bound\n");
    exit(1);
  }

  //Going to use a cyclic distribution
  int bound = atoi(argv[1]);
  for(int i=rank; i<=bound; i+=nprocs){
    if (i%1000000 == 0) {
      printf("i = %d\n", i);
      fflush(stdout);
    }
    if (is_perfect(i)) {
      printf("Found a perfect number: %d\n", i);
      num_perfect++;
    }
  }

  //Wait for all procs to finish finding their perfect numbers
  MPI_Barrier(MPI_COMM_WORLD);

  int totalCount = 0;
  //Here we need to consolidate all perfect numbers the procs have found
  MPI_Reduce(&num_perfect, &totalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if(rank==0){
    printf("Number of perfect numbers less than or equal to %d: %d.  Time = %lf\n",
	    bound, totalCount, mytime() - start_time);
  }

   MPI_Finalize();
}
