/* This program illustrates how the "private" clause affects
 * only references to the variable inside the _construct_
 * (the static extent), not the _region_ (dynamic extent).
 *
 * If you want x to be private everywhere, you need to use
 * the threadprivate directive.
 */
#include <stdio.h>
#include <omp.h>

int x = 99;

void f() {
  // this writes to the original x, not a private x...
  x=omp_get_thread_num();
}

int main() {
#pragma omp parallel private(x) num_threads(5)
  {
    int tid = omp_get_thread_num();

    f();
    printf("Thread %d: x = %d\n", tid, x);
  }
  printf("Final x = %d\n", x);
}
