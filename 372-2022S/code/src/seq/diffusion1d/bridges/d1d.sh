#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
# echo commands to stdout
set -x
./diffusion1d.exec 100000000 0.2 1000 1000 big.anim
