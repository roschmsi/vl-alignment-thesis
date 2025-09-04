#!/bin/bash
#SBATCH --job-name=download_images
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH -o /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_download_hf.out
#SBATCH -e /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_download_hf.err

source /dss/dsshome1/07/ga27tus3/miniconda3/etc/profile.d/conda.sh
conda activate sail_env_2

srun python /dss/dsshome1/07/ga27tus3/vision-language-alignment/data_preparation/download_huggingface.py