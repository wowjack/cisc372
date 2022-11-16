#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 00:01:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 10
set -x
mpirun -np $SLURM_NTASKS ./args.exec a b c
