#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]){
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	for(int i=0; i<10; i++){
		printf("%d: hi\n", rank);
		fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	for(int i=0; i<10; i++){
		printf("%d: bye\n", rank);
		fflush(stdout);
	}
	MPI_Finalize();
	return 0;
}
