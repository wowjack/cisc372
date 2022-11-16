NAME = pascal_mpi
NPROCS = 4 # default value.  Can be overridden on command line
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec Makefile
	$(MPIRUN) -n $(NPROCS) ./$(NAME).exec

$(NAME).exec: $(NAME).c Makefile
	$(MPICCC) -o $(NAME).exec $(NAME).c

.PHONY: all test
