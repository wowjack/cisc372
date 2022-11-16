NAME = datatype
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec
	$(MPIRUN) -n 2 ./$(NAME).exec

$(NAME).exec: $(NAME).c Makefile
	$(MPICCC) -o $@ $<

.PHONY: all test
