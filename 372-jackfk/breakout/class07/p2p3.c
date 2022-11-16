#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    //Get rank and comm size
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    //Seed and generate randoms
    srand(rank);
    int randnum = rand();
    //If not rank 0, send to rank 0 with tag of rank
    if(rank!=0){
        MPI_Send(&randnum, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
    }
    if(rank==0){
        printf("Process 0 has: %d\n", randnum);
        for(int i=1; i<nprocs; i++){
            MPI_Recv(&randnum, 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d has: %d\n", i, randnum);
        }
    }

    MPI_Finalize();
    return 0;
}