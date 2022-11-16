# Makefile for nbody graviational simulation, Pthread version
ROOT = ../../../../372-2022S/code
SNAME = nbody
NAME = $(SNAME)_pthread
SDIR = $(SEQ)/$(SNAME)
NCORES = 4
include $(ROOT)/common.mk
stems1 = planets-elliptical figure8 galaxy
stems =  $(stems1) galaxy2
movies1 = $(addsuffix .mp4,$(stems1))
movies = $(addsuffix .mp4,$(stems))
execs = $(addsuffix .exec,$(stems))
anims1 = $(addsuffix .anim,$(stems1))
anims = $(addsuffix .anim,$(stems))
tests1 = $(addsuffix .test,$(stems1))
tests = $(addsuffix .test,$(stems))

all: $(execs)

test:
	$(MAKE) -C $(SDIR) $(anims1)
	$(MAKE) $(tests1)

$(tests): %.test: %.anim
	$(RUN) diff $(SDIR)/$< $<

$(NAME).o: Makefile $(SDIR)/$(SNAME).h $(NAME).c $(ANIM)
	$(PTHREADCCO) -I$(SDIR) -O3 -o $(NAME).o $(NAME).c

$(execs): %.exec: $(SDIR)/%.c Makefile $(NAME).o $(ANIM)
	$(PTHREADCCC) -O3 -Wno-missing-braces -o $@ $< $(NAME).o -lanim -lm

$(anims): %.anim: %.exec
	$(PTHREADRUN) ./$< $(NCORES) $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

.PHONY: all test $(tests)
