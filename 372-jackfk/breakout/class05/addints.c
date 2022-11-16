#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char* argv[]){
	MPI_Init(&argc, &argv);
	unsigned long long localsum = 0;
	unsigned long long sum = 0;
	int rank, numProc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProc);
	for(int i=rank; i<=pow(10,9); i+=numProc){
		localsum+=i;
	}
	MPI_Reduce(&localsum, &sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	if(rank==0)printf("sum: %lld\n",sum);
	return 0;
}
