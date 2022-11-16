#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 00:01:00
#SBATCH -N 1
#SBATCH --gpus=1
# echo commands to stdout
set -x
./hello1.exec
./hello2.exec
./hello3.exec
