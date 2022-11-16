ROOT = ../../..
NCORES = 4
include $(ROOT)/common.mk
# The following require 0 command line args...
stems0 = adder dot hello1 hello2 hello3 initial initial2  \
         wait1 wait2_fix omp_workshare1 omp_workshare2 reduce sections \
         semiprivate sharedornot task threadprivate
# The following take some command line args or have errors...
stems1 = wait2 knights
stems = $(stems0) $(stems1)    
execs = $(addsuffix .exec,$(stems))

all: $(execs)

test: $(stems)

$(execs): %.exec: %.c Makefile
	$(OMPCCC) -o $@ $<

$(stems0): %: %.exec
	$(OMPRUN) ./$<

# This one requires some command line args...
knights: knights.exec
	$(OMPRUN) ./knights.exec a b c d e f g

# This one is supposed to have an error...
wait2: wait2.exec
	-$(OMPRUN) ./wait2.exec

.PHONY: $(stems) all test
