#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 00:01:00
#SBATCH -N 1
#SBATCH --gpus=1
set -x

cd ..
make complete_chaos.exec
cd bridges
../complete_chaos.exec complete_chaos.anim