NAME = max
ROOT = ../../code
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec
	$(CUDARUN) ./$<

$(NAME).exec: $(NAME).cu
	$(NVCCC) -o $@ $<

.PHONY: all test
