ROOT = ../../..
include $(ROOT)/common.mk
FLAGINC = $(INC)/pthread
FLAGSRC = $(SRC)/pthread/flag

all: spscsd.exec # mpmcmd.exec

test: spscsd # mpmcmd

spscsd.exec: spscsd.c $(FLAGINC)/flag.h
	$(PTHREADCCC) -I$(FLAGINC) -o spscsd.exec spscsd.c $(FLAGSRC)/flag.c

mpmcmd.exec: mpmcmd.c
	$(PTHREADCCC) -o mpmcmd.exec mpmcmd.c

spscsd: spscsd.exec
	./spscsd.exec

mpmcmd: mpmcmd.exec
	-timeout 2s ./mpmcmd.exec 10 3 3

