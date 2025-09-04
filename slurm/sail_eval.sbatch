#!/bin/bash
#SBATCH --job-name="evaluate alignment"
#SBATCH -p mcml-hgx-h100-94x4
#SBATCH --qos=mcml 
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH -o /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_eval_alignment.out
#SBATCH -e /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_eval_alignment.err

source /dss/dsshome1/07/ga27tus3/miniconda3/etc/profile.d/conda.sh
conda activate sail_env_2

bash /dss/dsshome1/07/ga27tus3/vision-language-alignment/scripts/sail_eval.sh