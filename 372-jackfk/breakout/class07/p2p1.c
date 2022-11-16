#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==1){
        int msgSend = 497;
        MPI_Send(&msgSend, 1, MPI_INT, 3, 50, MPI_COMM_WORLD);
    }
    if(rank==3){
        int msgrecv;
        MPI_Recv(&msgrecv, 1, MPI_INT, 1, 50, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received from proc 3: %d\n", msgrecv);
    }
    MPI_Finalize();
    return 0;
}