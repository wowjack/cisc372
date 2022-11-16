/* The integral of f(x)=1/(1+x^2) is atan(x).  So integral of f(x)
   from 0 to 1 is atan(1)-atan(0) = pi/4 - 0 = pi/4.  Therefore
   integral of 4f(x) from 0 to 1 is pi.  Let's estimate that integral
   by dividing the interval [0,1] into INTERVALS subintervals and
   adding the areas of the rectangles. */
#include<stdio.h>
#include <sys/time.h>
#define INTERVALS 1000000000L // 10^9 = 1 billion

double mytime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

int main() {
  const long double delta = 1.0L/(long double)INTERVALS,
    delta4 = 4.0L*delta;
  const double start_time = mytime();
  long double area = 0.0L;

  for (long i = 0; i < INTERVALS; i++) {
    const long double x = ((long double)i + 0.5L)*delta;
    area += delta4/(1.0L + x*x);
  }
  printf("Pi is %20.17Lf.  Time = %lf\n", area, mytime() - start_time);
}
