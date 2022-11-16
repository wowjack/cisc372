ROOT = ../../..
include $(ROOT)/common.mk

all: bank1.exec bank2.exec

test : bank1 bank2

bank1.exec : bank1.c Makefile
	$(PTHREADCCC) -o bank1.exec bank1.c

bank1 : bank1.exec
	-timeout 3s ./bank1.exec
	sleep 1

bank2.exec : bank2.c Makefile
	$(PTHREADCCC) -o bank2.exec bank2.c

bank2 : bank2.exec
	-timeout 4s ./bank2.exec 2 3 10
	sleep 1

.PHONY: bank1 bank2 all test
