#!/bin/bash
#SBATCH --job-name="encode"
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j.out
#SBATCH -e /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j.err

source /dss/dsshome1/07/ga27tus3/miniconda3/etc/profile.d/conda.sh
conda activate sail_env_2

bash /dss/dsshome1/07/ga27tus3/vision-language-alignment/scripts/encode.sh