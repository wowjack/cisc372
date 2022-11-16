ROOT = ../../..
include $(ROOT)/common.mk
stems = sat1 sat2 sat3
execs = $(addsuffix .exec,$(stems))

all: $(execs)

test: $(stems)

$(execs): %.exec: %.c Makefile
	$(MPICCC) -o $@ $<

$(stems): %: %.exec
	$(MPIRUN) -n 4 ./$<

.PHONY: $(stems) all test
