#!/bin/bash
# Reza Torbati
#
# Reasonable partitions: debug_5min, debug_30min, normal
#

#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
# memory in MB
#SBATCH --mem=5500
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=slurmOut/tunerV4Model4_analysis_stdout.txt
#SBATCH --error=slurmOut/tunerV4Model4_analysis_stderr.txt
#SBATCH --time=06:00:00
#SBATCH --job-name=MasteryOfPropulsion
#SBATCH --mail-user=eric.m.truong-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/etruong1014/symbiotic2/projects/baby_mocs/code_v2/tuner_emt
#SBATCH --array=0
#################################################

. /home/fagg/tf_setup.sh
conda activate tf_bleeding2

# Remove if using with sbatch
#export SLURM_ARRAY_TASK_ID=0

python3 post.py


