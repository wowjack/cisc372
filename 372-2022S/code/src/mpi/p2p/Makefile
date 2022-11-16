ROOT = ../../..
NPROCS = 4
include $(ROOT)/common.mk
good =  p2p status count tags anytag anysource may_deadlock cycle \
        shift exchange4 simplend wcdl_fix
stems = $(good) deadlock wcdl
execs = $(addsuffix .exec,$(stems))

all: $(execs)

test: $(good)

$(execs): %.exec: %.c Makefile
	$(MPICCC) -o $@ $<

$(stems): %: %.exec
	$(MPIRUN) -n $(NPROCS) ./$<

.PHONY: $(stems) all test
