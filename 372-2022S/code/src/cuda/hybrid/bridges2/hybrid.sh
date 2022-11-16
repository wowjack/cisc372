#!/bin/bash

# 1 node with 4 cores and 1 GPU is requested.
# On this node we run a program with 2 processes, each with 2 threads.
# Each process also invokes, once, a 3x4 GPU kernel.

#SBATCH -p GPU-shared
#SBATCH -t 00:01:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH --gpus=1
# echo commands to stdout
set -x
OMP_NUM_THREADS=2 mpirun -np 2 ./hybrid.exec
