NAME = diffusion1d
ROOT = ../../..
include $(ROOT)/common.mk
include $(NAME)_configs.mk
movies = d1.mp4 d2.mp4 d3.mp4 d4.mp4
anims = $(movies:.mp4=.anim)

all: $(NAME).exec

test: $(movies)

$(NAME).exec: $(NAME).c Makefile $(ANIM)
	$(CCC) -O3 -o $(NAME).exec $(NAME).c -lanim

$(anims): d%.anim: $(NAME).exec Makefile $(NAME)_configs.mk
	$(RUN) ./$(NAME).exec $(CONFIG$*) $@ 

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

big.anim: $(NAME).exec Makefile
	$(RUN) ./$(NAME).exec 100000000 0.2 1000 1000 big.anim

.PHONY: all test
