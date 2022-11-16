NAME = matmul
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec $(NAME)-dbg.exec

test: $(NAME)-dbg.exec Makefile
	$(RUN) ./$< 3 4 5

big: $(NAME).exec Makefile
	$(RUN) -p research ./$< 8000 8000 8000

$(NAME).exec: $(NAME).c Makefile
	$(CCC) -O3 -o $@ $<

$(NAME)-dbg.exec: $(NAME).c Makefile
	$(CCC) -DDEBUG -o $@ $<

.PHONY: all test big
