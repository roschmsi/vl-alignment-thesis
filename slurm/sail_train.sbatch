#!/bin/bash
#SBATCH --job-name="train alignment"
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --mem=900G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH -o /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_train_alignment.out
#SBATCH -e /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_train_alignment.err

source /dss/dsshome1/07/ga27tus3/miniconda3/etc/profile.d/conda.sh
conda activate sail_env_2

bash /dss/dsshome1/07/ga27tus3/vision-language-alignment/scripts/sail_train.sh