#include <stdio.h>

double f1(double x) {
  return x*x+1;
}

double f2(double x) {
  return x;
}

double addmeup(double f(double)) {
  double sum = 0;
  for (double i=0; i<10; i++)
    sum += f(i);
  return sum;
}

int main() {
  double s = addmeup(&f1);
  
  printf("addmeup of f1 = %f\n", s);
  printf("addmeup of f2 = %f\n", addmeup(&f2));
}
