NAME = matmul2
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec $(NAME)-dbg.exec

test: $(NAME)-dbg.exec
	$(CUDARUN) ./$< 3 4 5

big: $(NAME).exec
	$(CUDARUN) ./$< 8000 8000 8000

$(NAME).exec: $(NAME).cu Makefile
	$(NVCCC) -o $@ $<

$(NAME)-dbg.exec: $(NAME).cu Makefile
	$(NVCCC) -DDEBUG -o $@ $<

.PHONY: all test big
