NAME = perfect
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec Makefile
	$(RUN) ./$(NAME).exec 10
	$(RUN) ./$(NAME).exec 1000000

long: $(NAME).exec Makefile
	$(RUN) ./$(NAME).exec 10000000

$(NAME).exec: $(NAME).c Makefile
	$(CCC) -O3 -o $(NAME).exec $(NAME).c

.PHONY: all test long
