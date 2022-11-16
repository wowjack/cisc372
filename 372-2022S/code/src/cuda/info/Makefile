ROOT = ../../..
include $(ROOT)/common.mk
NAME = info

all: $(NAME).exec

test: $(NAME).out

$(NAME).out: $(NAME).exec
	$(CUDARUN) ./$(NAME).exec > $(NAME).out

$(NAME).exec: $(NAME).cu
	$(NVCCC) -o $@ $<

.PHONY: all test
