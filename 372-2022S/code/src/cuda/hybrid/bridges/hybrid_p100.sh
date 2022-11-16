#!/bin/bash

# A P100 node on the GPU-small partition has: 2 GPUs, 2 16-core CPUs,
# 8 TB on-node storage.  In this configuration, 1 node is requested.
# Two MPI processes will run, each will get 16 OpenMP threads.

#SBATCH -p GPU-small
#SBATCH -t 00:01:00
#SBATCH -N 1
#SBATCH --ntasks 2
#SBATCH --gres=gpu:p100:1
# echo commands to stdout
set -x
mpirun -np $SLURM_NTASKS ./hybrid.exec
