/* sat.c: simple SAT solver.
   Originally by Michael J. Quinn.  Updated by Stephen F. Siegel.
*/
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#define VARS 29

/* Return 1 if 'i'th bit of 'n' is 1; 0 otherwise */
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)

/* Returns true (1) if z satisfies the formula, else false (0) */
_Bool check_circuit(unsigned int z) {
  _Bool v[VARS];
  
  for (unsigned int i = 0; i < VARS; i++)
    v[i] = EXTRACT_BIT(z,i);
  if ((v[0] || v[1]) && (!v[1] || !v[3]) && (v[2] || v[3])
      && (!v[3] || !v[4]) && (v[4] || !v[5])
      && (v[5] || !v[6]) && (v[5] || v[6])
      && (v[6] || !v[15]) && (v[7] || !v[8])
      && (!v[7] || !v[13]) && (v[8] || v[9])
      && (v[8] || !v[9]) && (!v[9] || !v[10])
      && (v[9] || v[11]) && (v[10] || v[11])
      && (v[12] || v[13]) && (v[13] || !v[14])
      && (v[14] || v[15])
      && (v[16] || v[17]) && (!v[17] || !v[19]) && (v[18] || v[19])
      && (!v[19] || !v[20]) && (v[20] || !v[21])
      && (v[21] || !v[22]) && (v[21] || v[22])
      && (v[23] || !v[24])
      && (v[24] || v[25])
      && (v[24] || !v[25]) && (!v[25] || !v[26])
      && (v[25] || v[27]) && (v[26] || v[27])) {
    for (int j=0; j<VARS; j++) printf("%d", v[j]);
    printf("\n");
    fflush (stdout);
    return 1;
  }
  return 0;
}

int main() {
  int rank, nprocs;
  const unsigned int bound = 1u<<VARS; // 2^VARS
  unsigned int nsolutions = 0;
  
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (unsigned int i = rank; i < bound; i += nprocs)
    nsolutions += check_circuit(i);
  MPI_Finalize();
  printf("Proc %d: number of solutions found = %u\n", rank, nsolutions);
}
