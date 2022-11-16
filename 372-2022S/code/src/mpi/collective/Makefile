
ROOT = ../../..
include $(ROOT)/common.mk
stems = reduce allreduce bcast gather allgather scatter gatherv allgatherv \
        alltoall alltoallv alltoall_exercise emily
execs = $(addsuffix .exec,$(stems))

all: $(execs) block_exercise.exec

test: $(stems) block_exercise

$(execs): %.exec: %.c
	$(MPICCC) -o $@ $<

$(stems): %: %.exec
	$(MPIRUN) -n 5 ./$<

block_exercise.exec: Makefile block_exercise.c
	$(MPICCC) -o block_exercise.exec block_exercise.c

block_exercise: block_exercise.exec
	$(MPIRUN) -n 5 ./block_exercise.exec 13

.PHONY: $(stems) block_exercise all test
