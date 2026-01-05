#!/bin/bash
#SBATCH --job-name=generate_neighbor_indices
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH -o /home/eml/simon.roschmann/slurm/slurm-%j_generate_neighbor_idx.out
#SBATCH -e /home/eml/simon.roschmann/slurm/slurm-%j_generate_neighbor_idx.err

source /home/eml/simon.roschmann/miniconda3/etc/profile.d/conda.sh
conda activate ot_env

# srun python /home/eml/simon.roschmann/ot-alignment/generate_neighbor_indices.py \
# --query /lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/text_embedding/NV-Embed-v2/cc3m_raw_caption.h5 \
# --output /lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/text_embedding/NV-Embed-v2/cc3m_raw_caption_neighbors.npy

srun python /home/eml/simon.roschmann/ot-alignment/generate_neighbor_indices.py \
--query /lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/image_embedding/dinov2-large/cc3m_concat.h5 \
--output /lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/image_embedding/dinov2-large/cc3m_concat_neighbors.npy