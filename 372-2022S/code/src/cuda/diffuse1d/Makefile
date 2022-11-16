NAME = diffuse1d
VERSION = v1
ROOT = ../../..
include $(ROOT)/common.mk
CONFIGS = $(SEQ)/$(NAME)/$(NAME)_configs.mk
include $(CONFIGS)
texts = diffuse1.txt diffuse2.txt diffuse3.txt
tests = $(texts:.txt=.test)

all: $(NAME).exec

test:
	$(MAKE) -C $(SEQ)/$(NAME) $(texts)
	$(MAKE) $(tests)

$(tests): diffuse%.test: diffuse%.txt
	$(RUN) diff $(SEQ)/$(NAME)/$< $<

big.txt: $(NAME).exec
	$(CUDARUN) ./$(NAME).exec 10000000 0.2 1000 1000 big.txt

$(NAME).exec: $(NAME)_$(VERSION).cu Makefile
	$(NVCCC) -o $@ $<

$(texts): diffuse%.txt: $(NAME).exec Makefile $(CONFIGS)
	$(CUDARUN) ./$(NAME).exec $(CONF$*) $@

clean ::
	rm -f *.txt

.PHONY: all test $(tests) clean
