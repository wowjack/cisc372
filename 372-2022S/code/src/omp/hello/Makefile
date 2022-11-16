ROOT = ../../..
NCORES = 4
include $(ROOT)/common.mk
stems = hello_omp
execs = $(addsuffix .exec,$(stems))

all: $(execs) hello_mpiomp.exec

test: $(stems) mpiomp

$(execs): %.exec: %.c Makefile
	$(OMPCCC) -o $@ $<

$(stems): %: %.exec
	$(OMPRUN) ./$< 10

hello_mpiomp.exec: Makefile hello_mpiomp.c
	$(MPIOMPCCC) -o hello_mpiomp.exec hello_mpiomp.c

mpiomp: hello_mpiomp.exec
	$(MPIOMPRUN) -n 2 ./hello_mpiomp.exec 5

.PHONY: $(stems) all test mpiomp
