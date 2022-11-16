NAME = sat
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec Makefile
	$(RUN) time ./$(NAME).exec

$(NAME).exec: $(NAME).c Makefile
	$(CCC) -o $(NAME).exec $(NAME).c

.PHONY: all test
