#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    int numArgs;
    if(rank==0){
        numArgs = argc-1;
    }
    //Broadcast argc from proc 0 to all other procs
    MPI_Bcast(&numArgs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int args[numArgs];
    if(rank==0){
        for(int i=1; i<argc; i++){
            args[i-1] = atoi(argv[i]);
        }
    }
    MPI_Bcast(&args, numArgs, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Proc %d received: ", rank);
    for(int a=0; a<numArgs; a++){
        printf("%d ", args[a]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}