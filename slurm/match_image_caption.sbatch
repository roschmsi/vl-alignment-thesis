#!/bin/bash
#SBATCH --job-name=match
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_match.out
#SBATCH -e /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_match.err

source /dss/dsshome1/07/ga27tus3/miniconda3/etc/profile.d/conda.sh
conda activate sail_env_2

srun python /dss/dsshome1/07/ga27tus3/vision-language-alignment/data_preparation/match_image_caption.py --dataset cc12m