
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

char * dub(char * name) {
  char * new_name = malloc(strlen(name) + 5);
  memcpy(new_name, "Sir ", 4);
  strcpy(new_name + 4, name);
  return new_name;
}

int main(int argc, char * argv[]) {
  int n = argc - 1;
  if (n >= 1) {
    char * knights[n];
    #pragma omp parallel
    {
    int numThreads = omp_get_num_threads();
    int threadID = omp_get_thread_num();
    for (int i=threadID; i<n; i+=numThreads)
      knights[i] = dub(argv[i+1]);
    }
    for(int i=0; i<n; i++)
      printf("%s\n", knights[i]);
    #pragma omp parallel
    {
    int numThreads = omp_get_num_threads();
    int threadID = omp_get_thread_num();
    for (int i=threadID; i<n; i+=numThreads)
      free(knights[i]);
    }
  }
}