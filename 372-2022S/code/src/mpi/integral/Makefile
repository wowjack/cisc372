# Filename : Makefile
# Author   : Stephen F. Siegel, University of Delaware
# Date     : 16-apr-2020
# Makefile for MPI implementation of integral.
NAME = integral_mpi
NPROCS = 4 # default value.  Can be overridden on command line
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec Makefile
	-time $(MPIRUN) -n $(NPROCS) ./$(NAME).exec

$(NAME).exec: $(NAME).c Makefile
	$(MPICCC) -O3 -o $(NAME).exec $(NAME).c

.PHONY: all test
