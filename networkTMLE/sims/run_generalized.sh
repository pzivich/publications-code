#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=5g
#SBATCH -t 11-00:00
#SBATCH --array 10010-10015

python3 -u generalized_statin.py $SLURM_ARRAY_TASK_ID