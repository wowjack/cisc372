stems = pi_rect_mpi
execs = $(addsuffix .exec,$(stems))
NPROCS = 4
ROOT = ../../..
include $(ROOT)/common.mk

all: $(execs)

test: $(stems)

$(execs): %.exec: %.c
	$(MPICCC) -o $@ $<

$(stems): %: %.exec
	$(MPIRUN) -n $(NPROCS) ./$<

.PHONY: $(stems) all test
