#!/bin/bash
#SBATCH --job-name="encode"
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH -o /home/eml/simon.roschmann/slurm/slurm-%j_encode.out
#SBATCH -e /home/eml/simon.roschmann/slurm/slurm-%j_encode.err

source /home/eml/simon.roschmann/miniconda3/etc/profile.d/conda.sh
conda activate ot_env

vision_model="facebook/dinov2-large"
text_model="nvidia/NV-Embed-v2"
# data="imagenet1k"
# domain="image"
data="wikitext103"
domain="text"
batch_size=32
source_caption="raw_caption"
agg_mode="concat"
output_dir="/lustre/groups/eml/projects/sroschmann/ot-alignment"

python /home/eml/simon.roschmann/ot-alignment/encode.py \
--domain "$domain" \
--vision_model_name "$vision_model" \
--text_model_name "$text_model" \
--batch_size "$batch_size" \
--data "$data" \
--resume \
--source_caption "$source_caption" \
--agg_mode "$agg_mode" \
--output_dir "$output_dir" \
--num_workers 4
# --input_dir "$input_dir" \
# --start_shard_index 256
# --end_shard_index 255