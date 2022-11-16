/* In this example, every proc has a set of strings it wants to
   communicate, one string for every proc.  First, everyone has to
   find out the lengths of the incoming strings, using an Alltoall on
   ints.  Then they can allocate buffers to receive the incoming data
   with an Alltoallv, on chars.

   Apologies to Emily Dickinson,
   https://www.poemhunter.com/poem/nature-is-what-we-see/
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int rank, nprocs;
const int N = 23;
char * messages[] =
  {"Nature", "is", "what", "we", "see", "The Hill", "the Afternoon",
   "Squirrel", "Eclipse", "the Bumble bee", "Nay", "Nature is Heaven",
   "Nature is what we hear", "The Bobolink", "the Sea", "Thunder", "the Cricket",
   "Nay", "Nature is Harmony", "Nature is what we know", "Yet have no art to say",
   "So impotent Our Wisdom is", "To her Simplicity"};

int main() {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int sendcounts[nprocs], recvcounts[nprocs], sdispls[nprocs], rdispls[nprocs];
  int sbufsz = 0, rbufsz = 0;

  printf("Proc %d sending: ", rank);
  for (int i=0; i<nprocs; i++) {
    char * const msg = messages[(rank+i*nprocs)%N];
    const int nchars = strlen(msg) + 1; // +1 for null terminator char

    if (i>0) printf(" | ");
    printf("%s", msg);
    sdispls[i] = sbufsz;
    sendcounts[i] = nchars;
    sbufsz += nchars;
  }
  printf("\n"); fflush(stdout);
  MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
  for (int i=0; i<nprocs; i++) {
    rdispls[i] = rbufsz;
    rbufsz += recvcounts[i];
  }
  
  char sbuf[sbufsz], rbuf[rbufsz];
  
  for (int i=0; i<nprocs; i++)
    strcpy(sbuf + sdispls[i], messages[(rank+i*nprocs)%N]);
  MPI_Alltoallv(sbuf, sendcounts, sdispls, MPI_CHAR,
		rbuf, recvcounts, rdispls, MPI_CHAR, MPI_COMM_WORLD);
  printf("Proc %d received: ", rank);
  for (int i=0; i<nprocs; i++) {
    if (i > 0) printf(" | ");
    printf("%s", rbuf + rdispls[i]);
  }
  printf("\n"); fflush(stdout);
  MPI_Finalize();
}
