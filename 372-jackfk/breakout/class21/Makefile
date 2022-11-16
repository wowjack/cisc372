NAME = perfect
ROOT = ../../../372-2022S/code
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec
	$(CUDARUN) ./$< 50000000

$(NAME).exec: $(NAME).cu
	$(NVCCC) -o $@ $<

.PHONY: all test