NAME = diffusion2d
ROOT = ../../..
include $(ROOT)/common.mk
CONFIGS = $(NAME)_configs.mk
include $(CONFIGS)
movies = diffusion2d1.mp4 diffusion2d2.mp4
anims = $(movies:.mp4=.anim)

all: $(NAME).exec

test: $(movies)

$(NAME).exec: Makefile $(NAME).c $(ANIM)
	$(CCC) -O3 -o $(NAME).exec $(NAME).c -lanim

$(anims): $(NAME)%.anim: $(NAME).exec Makefile $(CONFIGS)
	$(RUN) ./$(NAME).exec $(CONFIG$*) $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

med.anim: $(NAME).exec Makefile
	$(RUN) ./$(NAME).exec 10000 10000 0.2 1000 1000 med.anim

big.anim: $(NAME).exec Makefile
	$(RUN) ./$(NAME).exec 20000 15000 0.2 1000 1000 big.anim

.PHONY: all test
