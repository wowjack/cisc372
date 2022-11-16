ROOT = ../../..
include $(ROOT)/common.mk
SEMLIBDIR = ../sem
SEMLIB = $(SEMLIBDIR)/sem.o

all: crit_sec_sem.exec crit_sec_mutex.exec bakery.exec

test : crit_sec_sem crit_sec_mutex bakery

$(SEMLIB) : $(INC)/pthread/sem.h $(SEMLIBDIR)/sem.c $(SEMLIBDIR)/Makefile
	$(MAKE) -C $(SEMLIBDIR)

crit_sec_sem.exec : Makefile crit_sec_sem.c $(SEMLIB)
	$(PTHREADCCC) -I$(INC)/pthread -o crit_sec_sem.exec crit_sec_sem.c $(SEMLIB)

crit_sec_sem : crit_sec_sem.exec
	./crit_sec_sem.exec 10

crit_sec_mutex.exec : Makefile crit_sec_mutex.c
	$(PTHREADCCC) -o crit_sec_mutex.exec crit_sec_mutex.c

crit_sec_mutex : crit_sec_mutex.exec
	./crit_sec_mutex.exec 100

bakery.exec : Makefile bakery.c
	$(PTHREADCCC) -o bakery.exec bakery.c

bakery : bakery.exec
	./bakery.exec 10

.PHONY: all test crit_sec_sem crit_sec_mutex bakery
