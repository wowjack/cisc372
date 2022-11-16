ROOT = ../../..
include $(ROOT)/common.mk

all: libsem.a

test: # TODO

$(ROOT)/bin:
	[ -d $(ROOT)/bin ] || mkdir $(ROOT)/bin

$(ROOT)/lib:
	[ -d $(ROOT)/lib ] || mkdir $(ROOT)/lib

libsem.a: Makefile sem.o
	rm -f libsem.a
	ar rsv libsem.a sem.o
	cp libsem.a $(ROOT)/lib

sem.o: $(INC)/pthread/sem.h sem.c Makefile
	$(PTHREADCCO) -I$(INC)/pthread -o sem.o sem.c

.PHONY: all test
