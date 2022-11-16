#include <stdatomic.h>
#include "flag.h"

void flag_init(flag_t * f, _Bool val) {
  if (val)
    atomic_flag_clear(f);
  else
    atomic_flag_test_and_set(f); // sequential consistency
}

void flag_destroy(flag_t * f) {}

void flag_raise(flag_t * f) {
#ifdef DEBUG
  _Bool old = atomic_flag_test_and_set(f);
  assert(old);
#endif
  atomic_flag_clear(f);
}

void flag_lower(flag_t * f) {
  while (atomic_flag_test_and_set(f)) ;
}
