#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==1){
        char msgSend[] = "Greetings to proc 3 from proc 1!";
        MPI_Send(&msgSend, strlen(msgSend)+1, MPI_CHAR, 3, 50, MPI_COMM_WORLD);
    }
    if(rank==3){
        char msgrecv[100];
        MPI_Recv(&msgrecv, 100, MPI_CHAR, 1, 50, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("%s\n", msgrecv);
    }

    MPI_Finalize();
    return 0;
}