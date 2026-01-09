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
# conda activate ot_env_nvembed
conda activate ot_env

# vision_model="valeoai/franca_vitl14"
# vision_model="valeoai/franca_vitg14"
vision_model="facebook/dinov2-large"
# vision_model="facebook/dinov3-vit7b16-pretrain-lvd1689m"
# vision_model="facebook/dinov3-vitl16-pretrain-lvd1689m"
text_model="nvidia/NV-Embed-v2"
# text_model="Qwen/Qwen3-Embedding-8B"
# text_model="nvidia/llama-embed-nemotron-8b"
# data="imagenet1k"
# data="diffusion_db"
# data="coco"
# coco_caption_index=4
data="cc3m"
domain="text"
batch_size=32
source_caption="raw_caption"
agg_mode="concat"
input_dir="/lustre/groups/eml/projects/sroschmann/cc3m" # _recaptioned
# input_dir="/lustre/groups/eml/datasets/coco"
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
--num_workers 4 \
--input_dir "$input_dir" \
--output_dir "$output_dir" \
--split validation
# --coco_caption_index "$coco_caption_index"
# --start_shard_index 256
# --end_shard_index 255