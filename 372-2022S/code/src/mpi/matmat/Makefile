NAME = matmat_mpi
ROOT = ../../..
NPROCS = 4
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec Makefile
	$(MPIRUN) -n $(NPROCS) ./$(NAME).exec

$(NAME).exec: $(NAME).c Makefile
	$(MPICCC) -o $(NAME).exec $(NAME).c

.PHONY: all test
