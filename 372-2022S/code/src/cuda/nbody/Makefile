NAME = nbody
ROOT = ../../..
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

$(execs): %.exec: %.c Makefile $(NAME).cu $(ANIM)
	$(NVCCC) -o $@ $< $(NAME).cu -lanim

$(anims): %.anim: %.exec
	$(CUDARUN) ./$< $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

.PHONY: all test $(tests)
