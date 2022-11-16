# Filename : Makefile
# Author   : Stephen F. Siegel, University of Delaware
# Date     : 16-apr-2020
# Makefile for sequential integral.
NAME = integral
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec Makefile
	- $(RUN) time ./$(NAME).exec

$(NAME).exec: $(NAME).c Makefile
	- $(CCC) -O3 -o $(NAME).exec $(NAME).c

.PHONY: all test
