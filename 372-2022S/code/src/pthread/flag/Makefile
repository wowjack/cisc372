ROOT = ../../..
include $(ROOT)/common.mk
FLAGINC = $(INC)/pthread

all: flag.o

test:

flag.o: $(FLAGINC)/flag.h flag.c
	$(PTHREADCCO) -I$(FLAGINC) -o flag.o flag.c

