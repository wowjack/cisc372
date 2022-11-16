# Makefile for nbody graviational simulation
ROOT = ../../..
include $(ROOT)/common.mk
stems1 = planets-elliptical figure8 galaxy
stems = $(stems1) galaxy2
movies1 = $(addsuffix .mp4,$(stems1))
movies = $(addsuffix .mp4,$(stems))
execs = $(addsuffix .exec,$(stems))
anims = $(addsuffix .anim,$(stems))

all: $(execs)

test: $(movies1)

nbody.o: Makefile nbody.h nbody.c $(ANIM)
	$(CCO) -O3 -o nbody.o nbody.c

$(execs): %.exec: %.c Makefile nbody.o $(ANIM)
	$(CCC) -O3 -Wno-missing-braces -o $@ $< nbody.o -lanim -lm

$(anims): %.anim: %.exec
	$(RUN) ./$< $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

.PHONY: all test
