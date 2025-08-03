#!/bin/bash
#SBATCH --job-name="encode"
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -o /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j.out
#SBATCH -e /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j.err

source /dss/dsshome1/07/ga27tus3/miniconda3/etc/profile.d/conda.sh
conda activate sail_env_2

bash scripts/encode.sh