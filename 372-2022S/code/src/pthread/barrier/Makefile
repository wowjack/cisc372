ROOT   = ../../..
NCORES = 2
NITER  = 100

include $(ROOT)/common.mk

# Pthreads implementations of flags and semaphores...
# FLAGINC = $(INC)/pthread
# FLAGSRC = $(SRC)/pthread/flag
# SEMINC  = $(INC)/pthread
# SEMSRC  = $(SRC)/pthread/sem

# C11 implementations of flags and semaphores...
FLAGINC = $(INC)/c11
FLAGSRC = $(SRC)/c11/flag
SEMINC  = $(INC)/c11
SEMSRC  = $(SRC)/c11/sem

# stems1 implement barrier.h and will be linked with barrier_driver
stems1 = counter_flag_barrier coordinator_barrier tree_barrier \
         dissem_barrier
# stems2 are similar to stems1 but have bugs and are expected to fail
stems2 = tree_barrier_bug

INCLUDES =  $(INC)/barrier.h $(FLAGINC)/flag.h $(SEMINC)/sem.h
SOURCES  = $(FLAGSRC)/flag.c $(SEMSRC)/sem.c

stems  = $(stems1) $(stems2)
execs1 = $(addsuffix .exec,$(stems1))
execs2 = $(addsuffix .exec,$(stems2))
execs  = $(execs1) $(execs2)

all: $(execs)

good: $(stems1)

bad: $(stems2)

test: good bad

barrier_driver.o: Makefile $(INC)/barrier.h barrier_driver.c
	$(PTHREADCCO) -DDEBUG -o barrier_driver.o barrier_driver.c

$(execs): %.exec: %.c Makefile barrier_driver.o $(INCLUDES) $(SOURCES)
	$(PTHREADCCC) -I$(FLAGINC) -I$(SEMINC) -o $@ $< barrier_driver.o $(SOURCES)

$(stems1): %: %.exec
	$(PTHREADRUN) ./$< $(NCORES) $(NITER)

$(stems2): %: %.exec
	@echo "Executing buggy barrier implementation..."
	-timeout 5s $(PTHREADRUN) ./$< $(NCORES) $(NITER)
	@echo "Done with buggy barrier execution."

.PHONY: all test $(stems)
