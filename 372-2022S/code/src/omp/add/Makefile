ROOT = ../../..
NCORES = 4
include $(ROOT)/common.mk
stems = add_omp
execs = $(addsuffix .exec,$(stems))

all: $(execs)

test: $(stems)

$(execs): %.exec: %.c Makefile
	$(OMPCCC) -o $@ $<

$(stems): %: %.exec
	$(OMPRUN) ./$< 10

.PHONY: $(stems) all test
