#!/bin/bash

#SBATCH -p RM
#SBATCH -t 00:05:00
#SBATCH -N 1
set -x

../galaxy2.exec 32 galaxy2.anim
