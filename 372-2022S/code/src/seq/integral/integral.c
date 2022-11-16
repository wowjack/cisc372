/* Filename : integral.c
   Author   : Stephen F. Siegel, University of Delaware
   Date     : 16-apr-2020

   Numerical integration of function of a single variable via adaptive
   quadrature.  This example uses the function 4/(1+x^2), which
   integrated from 0 to 1 yields pi.  If long doubles have 128 bits,
   this algorithm will converge to 18 digits after the decimal point,
   yielding 3.14159265358979323830.
   
   Given an interval [a,b] and a desired tolerance, let A1 be the area
   of the trapezoid with vertices (a,0), (b,0), (a,f(a)), (b,f(b)).
   Let c=(a+b)/2.  Let leftArea be the area of the trapezoid with
   vertices (a,0), (c,0), (a,f(a)), (c,f(c)), and rightArea be the
   area of the trapezoid with vertices (c,0), (b,0), (c,f(c)),
   (b,f(b)).  Let A2=leftArea+rightArea.  If |A1-A2|<=tolerance, the
   interval is considered "converged" and A2 is returned.  Otherwise,
   two recursive calls are made to compute the integral over [a,c] and
   over [c,b], each with half of the original tolerance, and the sum
   of the two results is returned. */
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>

_Static_assert(sizeof(long double)*CHAR_BIT >= 128, "Need 128 bit floats");

typedef long double T;
int count = 0; // number of calls to integrate

static inline T f(const T x) { return 4/(1+x*x); }

/* Given two points on x-axis, a < b, as well as fa=f(a) and fb=f(b),
   the area of the trapezoid defined by (a,0), (b,0), (a,f(a)),
   (b,f(b)), and some tolerance, computes an approximation to the
   integral from a to b that is within that tolerance of the exact
   result. */
static T integrate(const T a, const T b, const T fa, const T fb,
		   const T area, const T tolerance) {
  const T delta = b - a, c = a + delta/2, fc = f(c),
    leftArea = (fa+fc)*delta/4, rightArea = (fc+fb)*delta/4,
    area2 = leftArea + rightArea;

  if (tolerance == 0) {
    printf("Insufficent precision to obtain desired tolerance.\n");
    exit(1);
  }
  count++;
  if (fabsl(area2 - area) <= tolerance) return area2;
  return integrate(a, c, fa, fc, leftArea, tolerance/2) +
    integrate(c, b, fc, fb, rightArea, tolerance/2);
}

static T integral(const T a, const T b, const T tolerance) {
  count = 0;
  return integrate(a, b, f(a), f(b), (f(a)+f(b))*(b-a)/2, tolerance);
}

int main() {
  T result = integral(0, 1, 1e-18);

  printf("Number of intervals: %d\n", count);
  printf("Result: %4.20Lf\n", result);
}
