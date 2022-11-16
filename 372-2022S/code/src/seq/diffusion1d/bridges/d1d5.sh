#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
# echo commands to stdout
set -x
./diffusion1d.exec 320000 0.15 1000000 10000 big.anim
