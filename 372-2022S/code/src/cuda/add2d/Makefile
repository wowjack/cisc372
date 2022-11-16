NAME = add2d
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec
	$(CUDARUN) ./$< 20000 10000

$(NAME).exec: $(NAME).cu Makefile
	$(NVCCC) -o $@ $<

.PHONY: all test
