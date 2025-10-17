#!/bin/bash
#SBATCH --job-name=generate_mmap
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH -o /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_generate_mmap.out
#SBATCH -e /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_generate_mmap.err

source /dss/dsshome1/07/ga27tus3/miniconda3/etc/profile.d/conda.sh
conda activate sail_env_2

srun python /dss/dsshome1/07/ga27tus3/ot-alignment/data_preparation/generate_mmap.py