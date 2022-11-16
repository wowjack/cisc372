#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 00:01:00
#SBATCH -N 1
#SBATCH --gpus=1
# echo commands to stdout
set -x

../mandelbrot.exec 1200 400 mb.anim