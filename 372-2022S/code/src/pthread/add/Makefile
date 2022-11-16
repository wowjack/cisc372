# addmeup.c and addmeup2.c are here just to demonstrate the use of function types
# addmeup2.c shows how you can use typedef to make a function type easier to use

ROOT = ../../..
NCORES = 10
include $(ROOT)/common.mk
stems = addmeup addmeup2 add_pthread add_pthread_fix
execs = $(addsuffix .exec,$(stems))

all: $(execs)

test: $(stems) race norace

$(execs): %.exec: %.c Makefile
	$(PTHREADCCC) -o $@ $<

$(stems): %: %.exec
	$(PTHREADRUN) ./$< 20

race : add_pthread.exec Makefile
	for i in {1..1000}; do ./add_pthread.exec 20; done | sort -n | uniq -c

norace : add_pthread_fix.exec Makefile
	for i in {1..1000}; do ./add_pthread_fix.exec 20; done | sort -n | uniq -c

.PHONY: $(stems) all test race norace
