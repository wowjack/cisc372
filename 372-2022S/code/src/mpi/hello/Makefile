ROOT = ../../..
include $(ROOT)/common.mk
stems = hello_mpi hello_barrier hibye
execs = $(addsuffix .exec,$(stems))

all: $(execs)

test: $(stems)

$(execs): %.exec: %.c
	$(MPICCC) -o $@ $<

$(stems): %: %.exec
	$(MPIRUN) -n 10 ./$<

.PHONY: $(stems) all test
