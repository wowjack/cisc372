
#include "sem.h"

// TODO:
// P: atomic fetch and decrement.   If old value < 0: atomic increment, repeat.
// V: atomic add

void semaphore_init(semaphore *s, int val) {
  atomic_store(s, val);
}

void semaphore_destroy(semaphore *s) {
}

void semaphore_V(semaphore *s) {
  atomic_fetch_add(s, 1);
}

void semaphore_P(semaphore *s) {
  while (1) {
    int old = atomic_load(s);
    while (old > 0) {
      if (atomic_compare_exchange_weak(s, &old, old-1)) return;
    }
  }
}
