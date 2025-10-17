#!/bin/bash
#SBATCH --job-name="train alignment"
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH -o /home/eml/simon.roschmann/slurm/slurm-%j_train_alignment.out
#SBATCH -e /home/eml/simon.roschmann/slurm/slurm-%j_train_alignment.err

source /home/eml/simon.roschmann/miniconda3/etc/profile.d/conda.sh
conda activate ot_env

bash /home/eml/simon.roschmann/ot-alignment/scripts/sail_train.sh