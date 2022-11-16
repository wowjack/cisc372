NAME = diffusion1d_omp
ROOT = ../../../../372-2022S/code
NCORES = 4
include $(ROOT)/common.mk
CONFIGS = $(SEQ)/diffusion1d/diffusion1d_configs.mk
include $(CONFIGS)
movies = d1.mp4 d2.mp4 d3.mp4 d4.mp4
anims = $(movies:.mp4=.anim)
tests = $(movies:.mp4=.test)

all: $(NAME).exec

test:
	$(MAKE) -C $(SEQ)/diffusion1d $(anims)
	for i in {1..4}; do rm -f $(anims); $(MAKE) NCORES=$$i $(tests); done

$(tests): d%.test: d%.anim
	$(RUN) diff $(SEQ)/diffusion1d/$< $<

$(NAME).exec: Makefile $(NAME).c $(ANIM)
	$(OMPCCC) -O3 -o $(NAME).exec $(NAME).c -lanim

$(anims): d%.anim: $(NAME).exec Makefile $(CONFIGS)
	OMP_NUM_THREADS=$(NCORES) $(OMPRUN) ./$(NAME).exec $(CONFIG$*) $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

big.anim: $(NAME).exec Makefile
	OMP_NUM_THREADS=$(NCORES) $(OMPRUN) \
          ./$(NAME).exec 100000000 0.2 1000 1000 big.anim

.PHONY: all test $(tests)
