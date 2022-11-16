ROOT = ../../..
include $(ROOT)/common.mk

all: mytime.exec

test: mytime.exec
	./mytime.exec

mytime.exec: Makefile mytime.c
	$(CCC) -o mytime.exec mytime.c

.PHONY: all test
