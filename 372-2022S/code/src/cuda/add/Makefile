NAME = add
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec
	$(CUDARUN) ./$< 200000000

$(NAME).exec: $(NAME).cu
	$(NVCCC) -o $@ $<

.PHONY: all test
