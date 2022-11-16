# Makefile for nbody graviational simulation
ROOT = ../../../372-2022S/code
include $(ROOT)/common.mk
stems1 = planets-elliptical figure8 galaxy complete_chaos pulse_ring
stems = $(stems1) galaxy2
movies1 = $(addsuffix .mp4,$(stems1))
movies = $(addsuffix .mp4,$(stems))
execs = $(addsuffix .exec,$(stems))
anims = $(addsuffix .anim,$(stems))

dumb:
	echo $(NVCCO)

all: $(execs)

test: $(movies1)

nbody.o: Makefile nbody.h nbody.cu $(ANIM)
	nvcc -I../../../372-2022S/code/include -c -Xcompiler -fopenmp -O3 -o nbody.o nbody.cu

$(execs): %.exec: %.c Makefile nbody.o $(ANIM)
	nvcc  -I../../../372-2022S/code/include  -L../../../372-2022S/code/lib -lgomp -O3 -o $@ $< nbody.o -lanim -lm

$(anims): %.anim: %.exec
	$(CUDARUN) ./$< $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

.PHONY: all test
