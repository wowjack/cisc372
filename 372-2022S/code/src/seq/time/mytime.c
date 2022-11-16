/* This program demos how to measure wall time in a C program.
 * The timer used here is specified in sys/time.h.  This is not
 * part of the standard C library, but should nevertheless 
 * be portable across most unix-like operating systems.
 * The timer has microsecond precision (i.e., 10^-6 seconds).
 */
#include <sys/time.h>
#include <stdio.h>

double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

int main() {
  double t1 = mytime();

  unsigned int s = 0;
  for (int i=0; i<100000000; i++) s=s+1;

  double t2 = mytime();
  printf("Time = %lf seconds\n", t2-t1);
}

