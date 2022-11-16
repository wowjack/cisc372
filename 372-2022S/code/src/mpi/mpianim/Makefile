ROOT = ../../..
NPROCS = 8
include $(ROOT)/common.mk
sources = stripes1.c stripes2.c stripes3.c more_stripes.c
execs   = $(sources:.c=.exec)
anims   = $(sources:.c=.anim)
movies  = $(sources:.c=.mp4)

all: $(execs)

test: $(movies)

$(execs): %.exec: %.c Makefile
	$(MPICCC) -o $@ $< -lanim

stripes1.anim stripes2.anim stripes3.anim: %.anim: %.exec Makefile
	$(MPIRUN) -n $(NPROCS) ./$<

more_stripes.anim: more_stripes.exec Makefile
	$(MPIRUN) -n $(NPROCS) ./more_stripes.exec 500

$(movies): %.mp4: %.anim $(A2M)
	$(RUN) $(A2M) -keep -fps 30 $< -o $@

.PHONY: all test
