#!/bin/bash
#SBATCH --job-name=download_images
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH -o /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_download_images.out
#SBATCH -e /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_download_images.err

source /dss/dsshome1/07/ga27tus3/miniconda3/etc/profile.d/conda.sh
conda activate sail_env_2

srun python /dss/dsshome1/07/ga27tus3/SAIL/data_preparation/download_images.py \
--csv_path /dss/mcmlscratch/07/ga27tus3/cc3m_3long_3short_1raw_captions_url.csv \
--data_name cc3m \
--num_processes 64 \
--chunk_size 500 \
--output_dir /dss/mcmlscratch/07/ga27tus3