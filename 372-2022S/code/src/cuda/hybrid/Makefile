
NAME = hybrid
ROOT = ../../..
include $(ROOT)/common.mk
NPROCS = 2
NCORES = 4

all: $(NAME).exec

test: $(NAME).exec
	$(MPIRUN) -n $(NPROCS) -c $(NCORES) --gres=gpu:1 ./$<

$(NAME).exec: $(NAME).cu Makefile
	nvcc --compiler-options -fopenmp -o $@ --compiler-bindir mpic++ $<

.PHONY: all test
