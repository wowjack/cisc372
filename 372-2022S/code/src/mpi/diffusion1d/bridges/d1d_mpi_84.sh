#!/bin/bash
#SBATCH -p RM
#SBATCH -t 00:05:00
#SBATCH -N 3
#SBATCH --ntasks-per-node 28
# echo commands to stdout
set -x
mpirun -np $SLURM_NTASKS ./diffusion1d_mpi.exec 100000000 0.2 1000 1000 /pylon5/ccz3ahp/sfsiegel/big.anim
