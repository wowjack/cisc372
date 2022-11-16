ROOT = ../../..
include $(ROOT)/common.mk
stems = hello1 hello2 hello3
execs = $(addsuffix .exec,$(stems))

all: $(execs)

test: $(stems)

$(execs): %.exec: %.cu
	$(NVCCC) -o $@ $<

$(stems): %: %.exec
	$(CUDARUN) ./$<

.PHONY: $(stems) all test
