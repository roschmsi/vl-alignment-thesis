#!/bin/bash
#SBATCH --job-name=download_images
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH -o /home/eml/simon.roschmann/slurm/slurm-%j_download_hf.out
#SBATCH -e /home/eml/simon.roschmann/slurm/slurm-%j_download_hf.err

source /home/eml/simon.roschmann/miniconda3/etc/profile.d/conda.sh
conda activate ot_env

srun python /home/eml/simon.roschmann/ot-alignment/data_preparation/download_huggingface.py \
    --dataset cc3m \
    --output-dir /lustre/groups/eml/projects/sroschmann/