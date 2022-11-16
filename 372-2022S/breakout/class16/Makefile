NAME = diffuse1d_pthread
ROOT = ../../../372-2022S/code

NCORES = 4
include $(ROOT)/common.mk
CONFIGS = $(SEQ)/diffuse1d/diffuse1d_configs.mk
include $(CONFIGS)
texts = diffuse1.txt diffuse2.txt diffuse3.txt
tests = $(texts:.txt=.test)

# The directory containing the barrier implementations...
BAR = $(ROOT)/src/pthread/barrier

# Pthreads implementations of flags and semaphores...
# FLAGINC = $(INC)/pthread
# FLAGSRC = $(SRC)/pthread/flag
# SEMINC  = $(INC)/pthread
# SEMSRC  = $(SRC)/pthread/sem

# C11 implementations of flags and semaphores (faster)...
FLAGINC = $(INC)/c11
FLAGSRC = $(SRC)/c11/flag
SEMINC  = $(INC)/c11
SEMSRC  = $(SRC)/c11/sem

all: $(NAME).exec

test:
	$(MAKE) -C $(SEQ)/diffuse1d $(texts)
	$(MAKE) $(tests)

$(tests): diffuse%.test: diffuse%.txt
	$(RUN) diff $(SEQ)/diffuse1d/$< $<

$(NAME).exec: Makefile $(NAME).c
	$(PTHREADCCC) -o $(NAME).exec -I$(FLAGINC) $(NAME).c $(BAR)/coordinator_barrier.c $(FLAGSRC)/flag.c

$(texts): diffuse%.txt: $(NAME).exec Makefile $(CONFIGS)
	$(PTHREADRUN) ./$(NAME).exec $(CONF$*) $@

clean ::
	rm -f diffuse*.txt

.PHONY: all test $(tests) clean
