#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=10g
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 11-00:00
#SBATCH --array 1-200

module add python/3.9.6
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH
python3 -u single_scftmle.py "$SLURM_ARRAY_TASK_ID" "30" "5"