NAME = nbody
ROOT = ../../..
NCORES = 4
NPROCS = 2
include $(ROOT)/common.mk
SDIR = $(SEQ)/nbody
stems = planets-elliptical
movies = $(addsuffix .mp4,$(stems))
execs = $(addsuffix .exec,$(stems))
anims = $(addsuffix .anim,$(stems))
tests = $(addsuffix .test,$(stems))

all: $(execs)

test:
	$(MAKE) -C $(SDIR) $(anims)
	$(MAKE) $(tests)

$(tests): %.test: %.anim
	$(RUN) diff $(SDIR)/$< $<

$(execs): %.exec: $(SDIR)/%.c Makefile $(NAME).c $(ANIM)
	$(MPIOMPCCC) -Wno-missing-braces -o $@ $(SDIR)/$*.c $(NAME).c -lanim

$(anims): %.anim: %.exec
	$(MPIOMPRUN) -n $(NPROCS) ./$< $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

.PHONY: all test $(tests)
