#include <mpi.h>
#include <stdio.h>

//This definetly runs into deadlock somewhere. I didn't get to finish it by then end of class.

int check_prime(int a);

int rank, nprocs;

int main(int argc, char* argv[]){
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// rank 0 is manager, all others workers
	if(rank==0){
		int count = 0, send = 1;
		int activeWorkers = 0;
		int receive;
		MPI_Status recvStatus;
		for(int i=1; i<nprocs; i++){ //send an int to every worker proc
			MPI_Send(&send, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			send++;
		}
		while(count < 100){
			MPI_Recv(&receive, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recvStatus);
			
		}
	}else{
		int num, result;
		while(1){
			MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(num == -1) break;
			result = check_prime(num);
			MPI_Send(&num, 1, MPI_INT, 0, num, MPI_COMM_WORLD);
		}
		printf("Worker %d finished\n", rank);
	}



	MPI_Finalize();
	return 0;
}


int check_prime(int a){
	int c;
	for ( c = 2 ; c <= a - 1 ; c++ ){
      		if ( a%c == 0 ) return 0;
   	}
   	return 1;
}
