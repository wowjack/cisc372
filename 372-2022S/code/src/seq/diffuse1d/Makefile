NAME = diffuse1d
ROOT = ../../..
include $(ROOT)/common.mk
include $(NAME)_configs.mk
texts = diffuse1.txt diffuse2.txt diffuse3.txt

all: $(NAME).exec

test: $(texts)

$(NAME).exec: $(NAME).c Makefile 
	$(CCC) -O3 -o $(NAME).exec $(NAME).c

$(texts): diffuse%.txt: $(NAME).exec Makefile $(NAME)_configs.mk
	$(RUN) ./$(NAME).exec $(CONF$*) $@

big.txt: $(NAME).exec Makefile
	$(RUN) ./$(NAME).exec 10000000 0.2 1000 1000 big.txt

clean::
	rm -f diffuse*.txt big.txt

.PHONY: all test clean
