NAME = diffuse1d_mpi
ROOT = ../../..
NPROCS = 4
include $(ROOT)/common.mk
CONFIGS = $(SEQ)/diffuse1d/diffuse1d_configs.mk
include $(CONFIGS)
texts = diffuse1.txt diffuse2.txt diffuse3.txt
tests = $(texts:.txt=.test)

all: $(NAME).exec

test:
	$(MAKE) -C $(SEQ)/diffuse1d $(texts)
	for i in {1..4}; do rm -f $(texts); $(MAKE) NPROCS=$$i $(tests); done

$(tests): diffuse%.test: diffuse%.txt
	$(RUN) diff $(SEQ)/diffuse1d/$< $<

$(NAME).exec: Makefile $(NAME).c
	$(MPICCC) -o $(NAME).exec $(NAME).c

$(texts): diffuse%.txt: $(NAME).exec Makefile $(CONFIGS)
	$(MPIRUN) -n $(NPROCS) ./$(NAME).exec $(CONF$*) $@

clean ::
	rm -f *.txt

.PHONY: all test $(tests) clean
