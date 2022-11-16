#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 00:01:00
#SBATCH -N 1
#SBATCH --gpus=1
set -x

cd ..
make pulse_ring.exec
cd bridges
../pulse_ring.exec pulse_ring.anim