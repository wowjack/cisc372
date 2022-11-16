ROOT = ../../..
include $(ROOT)/common.mk
seqs = block1 glob2loc loc2glob     # the sequential programs
pars = block1_simp_mpi block1_mpi   # the parallel programs
seq_execs = $(addsuffix .exec,$(seqs))
par_execs = $(addsuffix .exec,$(pars))
stems = $(seqs) $(pars)
execs = $(seq_execs) $(par_execs)

all: $(seq_execs) $(par_execs)

test: $(seqs) $(pars)


$(seq_execs): %.exec: %.c Makefile
	$(CCC) -o $@ $<

$(par_execs): %.exec: %.c Makefile
	$(MPICCC) -o $@ $<

block1: block1.exec Makefile
	./block1.exec

glob2loc: glob2loc.exec Makefile
	./glob2loc.exec 10 3

loc2glob: loc2glob.exec Makefile
	./loc2glob.exec 10 3

# all of these should yield the same answer except the last (n=3),
# which should have an assertion violation
block1_simp_mpi: block1_simp_mpi.exec
	$(MPIRUN) -n 1 ./block1_simp_mpi.exec
	$(MPIRUN) -n 2 ./block1_simp_mpi.exec
	$(MPIRUN) -n 4 ./block1_simp_mpi.exec

block1_mpi: block1_mpi.exec
	$(MPIRUN) -n 1 ./block1_mpi.exec
	$(MPIRUN) -n 2 ./block1_mpi.exec
	$(MPIRUN) -n 3 ./block1_mpi.exec
	$(MPIRUN) -n 4 ./block1_mpi.exec

bigtests: $(par_execs)
	$(MPIRUN) -n 5 ./block1_simp_mpi.exec
	$(MPIRUN) -n 10 ./block1_simp_mpi.exec
	$(MPIRUN) -n 20 ./block1_simp_mpi.exec
	-$(MPIRUN) -n 3 ./block1_simp_mpi.exec
	$(MPIRUN) -n 5 ./block1_mpi.exec
	$(MPIRUN) -n 6 ./block1_mpi.exec
	$(MPIRUN) -n 7 ./block1_mpi.exec
	$(MPIRUN) -n 8 ./block1_mpi.exec
	$(MPIRUN) -n 9 ./block1_mpi.exec
	$(MPIRUN) -n 10 ./block1_mpi.exec
	$(MPIRUN) -n 37 ./block1_mpi.exec
