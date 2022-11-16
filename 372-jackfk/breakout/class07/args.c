#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /*
    RUNNING INTO DEADLOCK SOMEWHERE
    probably something to do with recv
    */

    int numargs;
    if(rank==0){
        numargs = argc-1;
        for(int i=1; i<nprocs; i++){
            //Get then send the number of command line args with tag 0
            MPI_Send(&numargs, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        for(int s=1; s<nprocs; s++){
            for(int i=0; i<numargs-1; i++){
            //Send all the command line args individually
                MPI_Send(&argv[i+1], strlen(argv[i+1])+1, MPI_CHAR, s, 1, MPI_COMM_WORLD);
            }
        }
    }
    if(rank!=0){
        //Receive the number of command line args from proc 0 with tag 0
        MPI_Recv(&numargs, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Proc %d: the %d args are ", rank, numargs);
        char recvbuf[256];
        for(int i=0; i<numargs; i++){
            //Recieve all the command line args individually
            MPI_Recv(&recvbuf, 256, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s ", recvbuf);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}