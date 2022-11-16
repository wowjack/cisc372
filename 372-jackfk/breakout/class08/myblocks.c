#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void sendAndReceiveNumArgs(int* numArgs, int numProcs, int argc, int procRank);
int getFirstOwned(int procRank, int arrSize, int numProcs);
int getNumOwned(int procRank, int arrSize, int numProcs);

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    int rank, nprocs, numArgs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    /*
    THIS LOOP PRINTS OUT ALL ARGS OTHER THAN THE COMMAND EXECUTED
    for(int i=1; i<argc; i++){
        printf("%s\n", argv[i]);
    }
    */
    sendAndReceiveNumArgs(&numArgs, nprocs, argc, rank);

    int arrSizeForProc = getNumOwned(rank, numArgs, nprocs);
    int* myArr[arrSizeForProc];
    if(rank==0){
        int argsAsInts[argc-1];
        for(int i=1; i<argc; i++){
            argsAsInts[i] = atoi(argv[i]); 
        }
        for(int i=1; i<numProcs; i++){
            //send each proc its subarray of the argsAsInts array
        }
    }else{

    }

    MPI_Finalize();
    return 0;
}

//Number of args is stored in numArgs for each proc
void sendAndReceiveNumArgs(int* numArgs, int numProcs, int argc, int procRank){
    if(procRank==0){
        //Send the number of command line arguments to every process
        *numArgs = argc-1;
        for(int i=1; i<numProcs; i++){
            MPI_Send(numArgs, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }else{
        MPI_Recv(numArgs, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int getFirstOwned(int procRank, int arrSize, int numProcs){
    return procRank * arrSize / numProcs;
}
int getNumOwned(int procRank, int arrSize, int numProcs){
    return getFirstOwned(procRank+1, arrSize, numProcs) - getFirstOwned(procRank, arrSize, numProcs);
}
