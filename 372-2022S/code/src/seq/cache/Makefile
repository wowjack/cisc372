
ROOT = ../../..
include $(ROOT)/common.mk

all: rowmaj.exec colmaj.exec

test: rowmaj colmaj

rowmaj.exec: rowmaj.c Makefile
	$(CCC) -o rowmaj.exec rowmaj.c

rowmaj: rowmaj.exec
	$(RUN) ./rowmaj.exec

colmaj.exec: colmaj.c Makefile
	$(CCC) -o colmaj.exec colmaj.c

colmaj: colmaj.exec
	$(RUN) ./colmaj.exec

.PHONY: all test rowmaj colmaj
