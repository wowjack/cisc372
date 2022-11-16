NAME = mandelbrot
ROOT = ../../..
include $(ROOT)/common.mk
include $(NAME)_configs.mk
movies = mb1.mp4 mb2.mp4 mb3.mp4 mb4.mp4
anims = $(movies:.mp4=.anim)

all: $(NAME).exec

test: mb1.mp4 mb2.mp4 mb3.mp4

$(NAME).exec: $(NAME).c Makefile $(ANIM)
	$(CCC) -O3 -o $(NAME).exec $(NAME).c -lanim

$(anims): mb%.anim: $(NAME).exec Makefile $(NAME)_configs.mk
	$(RUN) ./$(NAME).exec $(CONFIG$*) $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) -color gray $< -o $@

.PHONY: all test
