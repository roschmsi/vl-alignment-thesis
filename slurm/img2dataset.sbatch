#!/bin/bash
#SBATCH --job-name=img2dataset
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH -o /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_img2dataset.out
#SBATCH -e /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_img2dataset.err

source /dss/dsshome1/07/ga27tus3/miniconda3/etc/profile.d/conda.sh
conda activate sail_env_2

