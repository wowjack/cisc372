NAME = diffusion2d_omp
ROOT = ../../../../372-2022S/code
NCORES = 4
include $(ROOT)/common.mk
CONFIGS = $(SEQ)/diffusion2d/diffusion2d_configs.mk
include $(CONFIGS)
movies = diffusion2d1.mp4 diffusion2d2.mp4
anims = $(movies:.mp4=.anim)
tests = $(movies:.mp4=.test)

all: $(NAME).exec

test:
	$(MAKE) -C $(SEQ)/diffusion2d $(anims)
	for i in {1..4}; do rm -f $(anims); $(MAKE) NCORES=$$i $(tests); done

$(tests): diffusion2d%.test: diffusion2d%.anim
	$(RUN) diff $(SEQ)/diffusion2d/$< $<

$(NAME).exec: Makefile $(NAME).c $(ANIM)
	$(OMPCCC) -O3 -o $(NAME).exec $(NAME).c -lanim

$(anims): diffusion2d%.anim: $(NAME).exec Makefile $(CONFIGS)
	OMP_NUM_THREADS=$(NCORES) $(OMPRUN) ./$(NAME).exec $(CONFIG$*) $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

big.anim: $(NAME).exec Makefile
	OMP_NUM_THREADS=$(NCORES) $(OMPRUN) \
          ./$(NAME).exec 20000 15000 0.2 1000 1000 big.anim

.PHONY: all test $(tests)
