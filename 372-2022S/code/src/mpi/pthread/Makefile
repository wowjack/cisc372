NAME = hello_mpi_pthread
ROOT = ../../..
NPROCS = 2
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec
	$(MPIRUN) -n $(NPROCS) ./$(NAME).exec

# on machines using SLURM like cisc372 you can do this:
# $(MPIRUN) -n $(NPROCS) -c $(NCORES) ./$(NAME).exec
# this will give n procs each with c cores for a total of n*c cores

$(NAME).exec: $(NAME).c
	$(MPICCC) -o $(NAME).exec $(NAME).c -lpthread

.PHONY: all test
