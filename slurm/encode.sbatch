#!/bin/bash
#SBATCH --job-name="encode"
#SBATCH -p mcml-hgx-h100-94x4,mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH -o /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_encode.out
#SBATCH -e /dss/dsshome1/07/ga27tus3/slurm/logs/slurm-%j_encode.err

source /dss/dsshome1/07/ga27tus3/miniconda3/etc/profile.d/conda.sh
conda activate sail_env_2

vision_model="facebook/dinov2-large"
text_model="nvidia/NV-Embed-v2"
data="dreamclipcc12m"
domain="image"
batch_size=32
source_caption="raw_caption"
agg_mode="concat"
output_dir="/dss/mcmlscratch/07/ga27tus3"

python /dss/dsshome1/07/ga27tus3/vision-language-alignment/encode_wds.py \
--domain "$domain" \
--vision_model_name "$vision_model" \
--text_model_name "$text_model" \
--batch_size "$batch_size" \
--data "$data" \
--resume \
--source_caption "$source_caption" \
--agg_mode "$agg_mode" \
--output_dir "$output_dir" \
--output_hidden_states \
--downsample \
--num_workers 4 \
--start_shard_index 768
# --end_shard_index 767