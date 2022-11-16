NAME = wavemaker1d
ROOT = ../../..
include $(ROOT)/common.mk
include $(NAME)_configs.mk
movies = w1.mp4 w2.mp4
anims = $(movies:.mp4=.anim)

all: $(NAME).exec

test: $(movies)

big: $(NAME).exec
	./$(NAME).exec 60000 40000 300 0.005 50000 50000 big.anim

$(NAME).exec: $(NAME).c Makefile $(ANIM)
	$(CCC) -o $(NAME).exec $(NAME).c -lanim -lm

$(anims): w%.anim: $(NAME).exec Makefile $(NAME)_configs.mk
	$(RUN) ./$(NAME).exec $(CONFIG$*) $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

.PHONY: all test big
