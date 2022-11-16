NAME = wavemaker1d_omp
ROOT = ../../../../372-2022S/code
NCORES = 2
include $(ROOT)/common.mk
CONFIGS = $(SEQ)/wavemaker1d/wavemaker1d_configs.mk
include $(CONFIGS)
movies = w1.mp4 w2.mp4
anims = $(movies:.mp4=.anim)
tests = $(movies:.mp4=.test)

all: $(NAME).exec

test:
	$(MAKE) -C $(SEQ)/wavemaker1d $(anims)
	for i in {1..4}; do rm -f $(anims); $(MAKE) NCORES=$$i $(tests); done

big: $(NAME).exec
	OMP_NUM_THREADS=$(NCORES) $(OMPRUN) ./$(NAME).exec 60000 40000 300 0.005 50000 50000 big.anim

$(tests): w%.test: w%.anim
	$(RUN) diff $(SEQ)/wavemaker1d/$< $<

$(NAME).exec: Makefile $(NAME).c $(ANIM)
	$(OMPCCC) -o $(NAME).exec $(NAME).c -lanim -lm

$(anims): w%.anim: $(NAME).exec Makefile $(CONFIGS)
	OMP_NUM_THREADS=$(NCORES) $(OMPRUN) ./$(NAME).exec $(CONFIG$*) $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

.PHONY: all test $(tests) big
