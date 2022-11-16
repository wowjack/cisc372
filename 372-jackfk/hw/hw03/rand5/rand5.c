#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    //Get rank and comm size
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    //Seed and generate randoms
    srand(rank*5);
    int randnum[5] = {rand(), rand(), rand(), rand(), rand()};
    //If not rank 0, send to rank 0 with tag of rank
    if(rank!=0){
        MPI_Send(&randnum, 5, MPI_INT, 0, rank, MPI_COMM_WORLD);
    }
    if(rank==0){
        long int sums[5] = {0, 0, 0, 0, 0};
        printf("Received from proc 0: %d \t%d \t%d \t%d \t%d\n", randnum[0],randnum[1],randnum[2],randnum[3],randnum[4]);
        for(int i=0; i<5; i++) sums[i] += randnum[i];
        for(int i=1; i<nprocs; i++){
            MPI_Recv(&randnum, 5, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Received from proc %d: %d \t%d \t%d \t%d \t%d\n", i, randnum[0],randnum[1],randnum[2],randnum[3],randnum[4]);
            for(int i=0; i<5; i++) sums[i] += randnum[i];
        }
        printf("====================================================================================================\n");
        printf("Sums:\t\t      ");
        for(int i=0; i<5; i++) printf("%ld\t", sums[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
