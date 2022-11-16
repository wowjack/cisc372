#!/bin/bash
#SBATCH -p RM
#SBATCH -t 00:01:00
#SBATCH -N 2
#SBATCH --ntasks-per-node 28
# echo commands to stdout
set -x
cd /pylon5/ccz3ahp/sfsiegel
mpirun -np $SLURM_NTASKS /home/sfsiegel/372/code/src/mpi/mpianim/stripes1.exec
