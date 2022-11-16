NAME = pi_rect
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec
	$(RUN) ./$(NAME).exec

$(NAME).exec: $(NAME).c Makefile
	$(CCC) -o $(NAME).exec $(NAME).c

.PHONY: all test
