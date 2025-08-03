#!/bin/bash
#SBATCH --job-name=filter_images
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH -o /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_filter_images.out
#SBATCH -e /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_filter_images.err

source /dss/dsshome1/07/ga27tus3/miniconda3/etc/profile.d/conda.sh
conda activate sail_env_2

srun python /dss/dsshome1/07/ga27tus3/SAIL/data_preparation/filter.py \
--input_csv_file /dss/mcmlscratch/07/ga27tus3/cc3m_3long_3short_1raw_captions_url.csv \
--image_base_path /dss/mcmlscratch/07/ga27tus3 \
--chunk_size 5000