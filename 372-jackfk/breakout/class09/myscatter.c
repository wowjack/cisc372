#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


//Running into invalid buffer pointer in scatterv call


int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    int numArgs;
    if(rank==0){
        numArgs = argc-1;
    }
    //Broadcast the number of args to every proc
    MPI_Bcast(&numArgs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int args[numArgs];
    if(rank==0){
        for(int i=1; i<argc; i++){
            args[i-1] = atoi(argv[i]);
        }
    }
    int sendCounts[nprocs];
    for(int i=0; i<nprocs; i++){
        sendCounts[i] = ((i+1) * numArgs / nprocs) - (i * numArgs / nprocs);
    }

    int displacements[nprocs];
    for(int i=0; i<nprocs; i++){
        displacements[i] = i * numArgs / nprocs;
    }

    /*
    printf("Sendcounts: ");
    for(int i=0; i<nprocs; i++){
        printf("%d ",sendCounts[i]);
    }
    printf("\n");

    return 0;
    */

    //Scatter the block distributed args to every proc
    MPI_Scatterv(&args, sendCounts, displacements, MPI_INT, &args, sendCounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    printf("Proc %d received: ", rank);
    for(int a=0; a<numArgs; a++){
        printf("%d ", args[a]);
    }
    printf("\n");


    MPI_Finalize();
    return 0;
}