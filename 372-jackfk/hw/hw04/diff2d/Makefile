ROOT = ../../../../372-2022S/code
NPROCS = 6
include $(ROOT)/common.mk

all: diff2d_mpi.exec

diff2d_mpi.exec: diff2d_mpi.c
	$(MPICCC) -o diff2d_mpi.exec diff2d_mpi.c -lanim

diff2d_mpi.anim: diff2d_mpi.exec
	$(MPIRUN) -n $(NPROCS) ./diff2d_mpi.exec

diff2d_mpi.mp4: diff2d_mpi.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

test: diff2d_mpi.anim diff2d.anim
	$(RUN) diff diff2d.anim diff2d_mpi.anim

diff2d.mp4: diff2d.anim $(A2M)
	$(RUN) $(A2M) $< -o $@

diff2d.anim: diff2d.exec
	$(RUN) ./diff2d.exec

diff2d.exec: diff2d.c
	$(CCC) -o diff2d.exec diff2d.c -lanim

.PHONY: all test
