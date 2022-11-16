
ROOT = ../../..
include $(ROOT)/common.mk

all: hello.exec

test: hello

hello.exec: Makefile hello.c
	$(CCC) -o hello.exec hello.c

hello: hello.exec
	./hello.exec

.PHONY: all test hello
