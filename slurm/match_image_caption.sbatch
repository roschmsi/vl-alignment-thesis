#!/bin/bash
#SBATCH --job-name=match_image_caption
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -o /home/eml/simon.roschmann/slurm/slurm-%j_match_images_caption.out
#SBATCH -e /home/eml/simon.roschmann/slurm/slurm-%j_match_images_caption.err

source /home/eml/simon.roschmann/miniconda3/etc/profile.d/conda.sh
conda activate ot_env

srun python /home/eml/simon.roschmann/ot-alignment/data_preparation/match_image_caption.py \
    --dataset cc3m \
    --base_dir /lustre/groups/eml/projects/sroschmann