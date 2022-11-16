NAME = diffusion2d
ROOT = ../../..
include $(ROOT)/common.mk
CONFIGS = $(SEQ)/$(NAME)/$(NAME)_configs.mk
include $(CONFIGS)
movies = diffusion2d1.mp4 diffusion2d2.mp4
anims = $(movies:.mp4=.anim)
tests = $(movies:.mp4=.test)

all: $(NAME).exec

test:
	$(MAKE) -C $(SEQ)/$(NAME) $(anims)
	$(MAKE) $(tests)

$(tests): $(NAME)%.test: $(NAME)%.anim
	$(RUN) diff $(SEQ)/$(NAME)/$< $<

$(NAME).exec: $(NAME).cu $(ANIM) Makefile
	$(NVCCC) --fmad false -o $@ $< -lanim

$(anims): $(NAME)%.anim: $(NAME).exec Makefile $(CONFIGS)
	$(CUDARUN) ./$(NAME).exec $(CONFIG$*) $@

med.anim: $(NAME).exec Makefile
	$(CUDARUN) ./$(NAME).exec 10000 10000 0.2 1000 1000 med.anim

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

.PHONY: all test $(tests)
