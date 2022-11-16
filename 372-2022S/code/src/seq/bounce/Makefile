NAME = bounce
ROOT = ../../..
include $(ROOT)/common.mk
CONFIGS = $(NAME)_configs.mk
include $(CONFIGS)
movies = bounce1.mp4 bounce2.mp4
anims = $(movies:.mp4=.anim)

all: $(NAME).exec

test: $(movies)

$(NAME).exec: Makefile $(NAME).c $(ANIM)
	$(CCC) -O3 -o $(NAME).exec $(NAME).c -lanim

$(anims): $(NAME)%.anim: $(NAME).exec Makefile $(CONFIGS)
	$(RUN) ./$(NAME).exec $(CONFIG$*) $@

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@ -fps 60

.PHONY: all test
