NAME = args
NPROCS = 10
ROOT = ../../..
include $(ROOT)/common.mk

all: $(NAME).exec

test: $(NAME).exec Makefile
	$(MPIRUN) -n $(NPROCS) ./$(NAME).exec a b c

$(NAME).exec: $(NAME).c Makefile
	$(MPICCC) -o $(NAME).exec $(NAME).c

