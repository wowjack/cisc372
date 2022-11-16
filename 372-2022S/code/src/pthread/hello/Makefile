ROOT = ../../..
include $(ROOT)/common.mk
stems = hello_pthread
execs = $(addsuffix .exec,$(stems))
NCORES=10

all: $(execs)

test: $(stems)

$(execs): %.exec: %.c Makefile
	$(PTHREADCCC) -o $@ $<

$(stems): %: %.exec
	$(PTHREADRUN) ./$< 10

.PHONY: $(stems) all test
