ROOT = ../../..
NCORES = 4
include $(ROOT)/common.mk
# Pthreads implementations of flags...
# FLAGINC = $(INC)/pthread
# FLAGSRC = $(SRC)/pthread/flag
# C11 implementations of flags...
FLAGINC = $(INC)/c11
FLAGSRC = $(SRC)/c11/flag

all: 2barrier.exec 2barrier_bug.exec

2barrier.exec 2barrier_bug.exec: %.exec : %.c Makefile \
     $(FLAGINC)/flag.h $(FLAGSRC)/flag.c
	$(PTHREADCCC) -I$(FLAGINC) -o $@ $< $(FLAGSRC)/flag.c

test: 2barrier.exec
	$(PTHREADRUN) ./2barrier.exec

test_bug: 2barrier_bug.exec
	$(PTHREADRUN) ./2barrier_bug.exec

.PHONY: all test test_bug
