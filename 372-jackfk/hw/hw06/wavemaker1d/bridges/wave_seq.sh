#!/bin/bash

#SBATCH -p RM
#SBATCH -t 00:05:00
#SBATCH -N 1
set -x

../wavemaker1d_omp.exec 20000000 10000 1000 0.005 1000 1000 out.anim
