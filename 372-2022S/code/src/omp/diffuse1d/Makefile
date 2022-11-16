NAME = diffuse1d_omp
ROOT = ../../..
NCORES = 4
include $(ROOT)/common.mk
CONFIGS = $(SEQ)/diffuse1d/diffuse1d_configs.mk
include $(CONFIGS)
texts = diffuse1.txt diffuse2.txt diffuse3.txt
tests = $(texts:.txt=.test)

all: $(NAME).exec

test:
	$(MAKE) -C $(SEQ)/diffuse1d $(texts)
	for i in {1..4}; do rm -f $(texts); $(MAKE) NCORES=$$i $(tests); done

$(tests): diffuse%.test: diffuse%.txt
	$(RUN) diff $(SEQ)/diffuse1d/$< $<

$(NAME).exec: Makefile $(NAME).c
	$(OMPCCC) -o $(NAME).exec $(NAME).c

$(texts): diffuse%.txt: $(NAME).exec Makefile $(CONFIGS)
	OMP_NUM_THREADS=$(NCORES) $(OMPRUN) ./$(NAME).exec $(CONF$*) $@

clean::
	rm -f diffuse*.txt

.PHONY: all test $(tests)
