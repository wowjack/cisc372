ROOT = ../../..
include $(ROOT)/common.mk
FLAGINC = $(INC)/c11

all: flag.o

test:

flag.o: $(FLAGINC)/flag.h flag.c
	$(CCO) -I$(FLAGINC) -o flag.o flag.c

