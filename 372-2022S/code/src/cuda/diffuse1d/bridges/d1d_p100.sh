#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:p100:1
# echo commands to stdout
set -x
./diffuse1d.exec 10000000 0.2 1000 1000 big.txt
