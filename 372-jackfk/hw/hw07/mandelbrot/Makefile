NAME = mandelbrot
ROOT = ../../../../372-2022S/code
include $(ROOT)/common.mk
CONFIGS = $(SEQ)/mandelbrot/mandelbrot_configs.mk
include $(CONFIGS)
movies = mb1.mp4 mb2.mp4 mb3.mp4 mb4.mp4
anims = $(movies:.mp4=.anim)
tests = $(movies:.mp4=.test)

all: $(NAME).exec

test:
	$(MAKE) -C $(SEQ)/mandelbrot $(anims)
	$(MAKE) $(tests)

$(tests): mb%.test: mb%.anim
	$(RUN) diff $(SEQ)/mandelbrot/$< $<

$(NAME).exec: $(NAME).cu $(ANIM) Makefile
	$(NVCCC) --fmad false -o $@ $< -lanim

$(anims): mb%.anim: $(NAME).exec Makefile $(CONFIGS)
	$(CUDARUN) ./$(NAME).exec $(CONFIG$*) $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

.PHONY: all test $(tests)
